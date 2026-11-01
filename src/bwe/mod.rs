//! Google Congestion Control (GoogCC) Bandwidth Estimation based on TWCC feedback.
//!
//! This implementation is ported from libWebRTC's GoogCC and goes beyond the simplified
//! IETF draft (<https://datatracker.ietf.org/doc/html/draft-ietf-rmcat-gcc-02>) to include
//! WebRTC's production features:
//!
//! - Delay-based control (trendline estimator with AIMD rate control)
//! - Loss-based control (with inherent loss rate estimation)
//! - Probe controller with state machine and multi-stage probing strategy
//! - ALR (Application Limited Region) detection and periodic probing
//! - Link capacity estimation from ALR probes
//!
//! The probe controller in particular closely matches WebRTC's `ProbeController` behavior
//! and default constants, enabling compatible bandwidth discovery with WebRTC endpoints.

use std::cmp::Ordering;
use std::fmt;
use std::time::{Duration, Instant};

use crate::Reason;
use crate::rtp_::{Bitrate, DataSize, TwccClusterId, TwccSendRecord, TwccSeq};
use crate::util::Soonest;
use crate::util::value_history::ValueHistory;

mod acked_bitrate_estimator;
mod alr_detector;
pub(crate) mod api;
mod delay;
mod link_capacity_estimator;
mod loss_controller;
mod macros;
mod probe;
mod smoother;
mod time;

use acked_bitrate_estimator::AckedBitrateEstimator;
use alr_detector::AlrDetector;
use delay::DelayController;
use link_capacity_estimator::LinkCapacityEstimator;
use loss_controller::{LossController, LossControllerState, PacketResult};
use macros::log_loss;
use smoother::EstimateSmoother;

pub(crate) use macros::{log_pacer_media_debt, log_pacer_padding_debt};
pub(crate) use probe::{BandwidthLimitedCause, ProbeEstimator};
pub(crate) use probe::{ProbeClusterState, ProbeControl};

#[cfg(feature = "_internal_test_exports")]
pub use probe::ProbeClusterConfig;
#[cfg(not(feature = "_internal_test_exports"))]
pub(crate) use probe::ProbeClusterConfig;

const INITIAL_BITRATE_WINDOW: Duration = Duration::from_millis(500);
const BITRATE_WINDOW: Duration = Duration::from_millis(150);
const STARTUP_PHASE: Duration = Duration::from_secs(2);

/// Data channel transmissions held for the next transport feedback report.
/// Media keeps reports flowing, so this only bounds a session without them.
const MAX_PENDING_DATA_CHANNEL_SAMPLES: usize = 4096;

/// The window over which acknowledged data channel bytes count as sustained.
const DATA_CHANNEL_ACKED_WINDOW: Duration = Duration::from_secs(1);

/// A data channel DATA chunk acknowledged by the peer's SCTP stack.
///
/// TWCC covers RTP only, so these are the only loss signal for data channel
/// traffic sharing the path with media.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DataChannelDelivery {
    /// When the chunk was first sent.
    pub first_sent: Instant,
    /// When the chunk was last sent, the transmission that arrived.
    pub last_sent: Instant,
    /// When the acknowledgement arrived.
    pub acked_at: Instant,
    /// User data carried by each transmission.
    pub size: DataSize,
    /// Transmissions that did not reach the peer.
    pub lost_transmissions: u32,
}

/// One transmission and whether it arrived, from TWCC or SCTP.
#[derive(Debug, Clone, Copy)]
struct LossSample {
    send_time: Instant,
    size: DataSize,
    lost: bool,
}

impl PacketResult for LossSample {
    fn local_send_time(&self) -> Instant {
        self.send_time
    }

    fn size(&self) -> DataSize {
        self.size
    }

    fn lost(&self) -> bool {
        self.lost
    }
}

impl From<&TwccSendRecord> for LossSample {
    fn from(record: &TwccSendRecord) -> Self {
        LossSample {
            send_time: record.local_send_time(),
            size: record.size().into(),
            lost: record.remote_recv_time().is_none(),
        }
    }
}

pub struct Bwe {
    bwe: SendSideBandwidthEstimator,
    desired_bitrate: Bitrate,
    smoother: EstimateSmoother,
}

impl Bwe {
    pub fn new(initial: Bitrate) -> Self {
        let send_side_bwe = SendSideBandwidthEstimator::new(initial);
        Bwe {
            bwe: send_side_bwe,
            desired_bitrate: Bitrate::ZERO,
            smoother: EstimateSmoother::new(),
        }
    }

    pub fn handle_timeout(&mut self, now: Instant, do_probe: bool) -> Option<ProbeClusterConfig> {
        let result = self.bwe.handle_timeout(self.desired_bitrate, do_probe, now);
        if let Some(estimate) = self.bwe.last_estimate() {
            self.smoother.record(now, estimate);
        }
        result
    }

    pub fn start_probe(&mut self, config: ProbeClusterConfig, now: Instant) -> bool {
        self.bwe.start_probe(config, now)
    }

    pub fn end_probe(&mut self, now: Instant, cluster_id: TwccClusterId) {
        self.bwe.end_probe(now, cluster_id);
    }

    pub fn reset(&mut self, init_bitrate: Bitrate) {
        self.bwe.reset(init_bitrate);
    }

    pub fn update<'t>(
        &mut self,
        records: impl Iterator<Item = &'t crate::rtp_::TwccSendRecord>,
        now: Instant,
    ) {
        self.bwe.update(records, now);
    }

    pub fn poll_estimate(&mut self) -> Option<Bitrate> {
        self.smoother.poll()
    }

    pub fn poll_timeout(&self) -> (Option<Instant>, Reason) {
        self.bwe.poll_timeout()
    }

    pub fn last_estimate(&self) -> Option<Bitrate> {
        self.bwe.last_estimate()
    }

    pub fn on_media_sent(&mut self, payload_size: DataSize, is_padding: bool, now: Instant) {
        if !is_padding {
            // Update ALR detector with media bytes sent
            self.bwe.on_media_sent(payload_size, now);
        }
    }

    pub fn on_data_channel_delivery(&mut self, delivery: DataChannelDelivery) {
        self.bwe.on_data_channel_delivery(delivery);
    }

    pub fn is_overusing(&self) -> bool {
        self.bwe.is_overusing()
    }

    pub fn set_desired_bitrate(&mut self, v: Bitrate) {
        self.desired_bitrate = v;
    }
}

struct SendSideBandwidthEstimator {
    delay_controller: DelayController,
    loss_controller: LossController,
    acked_bitrate_estimator: AckedBitrateEstimator,
    probe_control: ProbeControl,
    probe_estimator: ProbeEstimator,
    started_at: Option<Instant>,
    alr_detector: AlrDetector,
    link_capacity_estimator: LinkCapacityEstimator,
    last_updated_estimate: Option<Bitrate>,
    /// Data channel transmissions resolved since the last TWCC report.
    data_channel_samples: Vec<LossSample>,
    /// Data channel bytes acknowledged within the sustained window.
    data_channel_acked: ValueHistory<DataSize>,
    /// Latest data channel acknowledgement time pushed into the history.
    data_channel_acked_at: Option<Instant>,
}

impl SendSideBandwidthEstimator {
    pub fn new(initial_bitrate: Bitrate) -> Self {
        let mut alr_detector = AlrDetector::new();
        alr_detector.set_estimated_bitrate(initial_bitrate);

        let mut loss_controller = LossController::new();
        loss_controller.set_bandwidth_estimate(initial_bitrate);

        Self {
            delay_controller: DelayController::new(initial_bitrate),
            loss_controller,
            acked_bitrate_estimator: AckedBitrateEstimator::new(
                INITIAL_BITRATE_WINDOW,
                BITRATE_WINDOW,
            ),
            probe_control: ProbeControl::new(),
            probe_estimator: ProbeEstimator::new(),
            started_at: None,
            alr_detector,
            link_capacity_estimator: LinkCapacityEstimator::new(),
            last_updated_estimate: None,
            data_channel_samples: Vec::new(),
            data_channel_acked: ValueHistory::default(),
            data_channel_acked_at: None,
        }
    }

    /// Record a data channel delivery for the loss and throughput estimates.
    ///
    /// Each transmission counts as one packet: the lost ones at the first
    /// send time, which SCTP knows exactly for the first loss, and the one
    /// that arrived at the last send time.
    pub fn on_data_channel_delivery(&mut self, delivery: DataChannelDelivery) {
        let lost = (0..delivery.lost_transmissions).map(|_| LossSample {
            send_time: delivery.first_sent,
            size: delivery.size,
            lost: true,
        });
        let arrived = LossSample {
            send_time: delivery.last_sent,
            size: delivery.size,
            lost: false,
        };
        self.data_channel_samples
            .extend(lost.chain(std::iter::once(arrived)));
        let excess = self
            .data_channel_samples
            .len()
            .saturating_sub(MAX_PENDING_DATA_CHANNEL_SAMPLES);
        self.data_channel_samples.drain(..excess);

        // A report held for loss evidence can carry an earlier ACK time.
        let acked_at = self
            .data_channel_acked_at
            .map_or(delivery.acked_at, |t| t.max(delivery.acked_at));
        self.data_channel_acked_at = Some(acked_at);
        self.data_channel_acked.push(acked_at, delivery.size);
    }

    /// Data channel throughput acknowledged over the sustained window.
    fn data_channel_acked_bitrate(&mut self, now: Instant) -> Bitrate {
        self.data_channel_acked.purge_old(now);
        self.data_channel_acked.sum() / DATA_CHANNEL_ACKED_WINDOW
    }

    /// Everything the path acknowledged: RTP from TWCC plus data channels.
    fn total_acked_bitrate(&mut self, rtp: Option<Bitrate>, now: Instant) -> Option<Bitrate> {
        let data_channel = self.data_channel_acked_bitrate(now);
        match rtp {
            Some(rtp) => Some(rtp + data_channel),
            None => (data_channel > Bitrate::ZERO).then_some(data_channel),
        }
    }

    /// Whether the delay-based detector currently signals overuse.
    ///
    /// This is useful for gating behaviors (like padding/probing) that would otherwise
    /// re-excite the system while we're already congested.
    pub fn is_overusing(&self) -> bool {
        self.delay_controller.is_overusing()
    }

    /// Update ALR detector with actual bytes sent.
    ///
    /// Should be called for media packets (not padding/probes).
    /// This is typically called from the session's packet sending logic.
    pub fn on_media_sent(&mut self, bytes: DataSize, now: Instant) {
        self.alr_detector.on_bytes_sent(bytes, now);
    }

    /// Record a packet from a TWCC report.
    pub fn update<'t>(&mut self, records: impl Iterator<Item = &'t TwccSendRecord>, now: Instant) {
        let _ = self.started_at.get_or_insert(now);

        let send_records: Vec<_> = records.collect();

        // Feed records to probe estimator for analysis and process any new probe results
        let mut latest_probe_result = None;
        for (config, bitrate) in self.probe_estimator.update(send_records.iter().copied()) {
            latest_probe_result = Some(bitrate);

            // Update link capacity estimator for every successful ALR probe, not just the latest.
            // The estimator internally takes the max of all probe results, building up knowledge
            // of proven link capacity. This differs from the delay controller, which only receives
            // the latest probe result (matching WebRTC's FetchAndResetLastEstimatedBitrate behavior).
            if config.is_alr_probe() {
                self.link_capacity_estimator.update_from_probe(bitrate, now);
            }
        }

        let mut acked_packets = vec![];

        let mut max_rtt = None;
        let mut count = 0;
        let mut lost = 0;
        for record in send_records.iter() {
            count += 1;
            let Ok(acked_packet) = (*record).try_into() else {
                lost += 1;
                continue;
            };
            acked_packets.push(acked_packet);
            max_rtt = max_rtt.max(record.rtt());
        }
        acked_packets.sort_by(AckedPacket::order_by_receive_time);

        for acked_packet in acked_packets.iter() {
            self.acked_bitrate_estimator
                .update(acked_packet.remote_recv_time, acked_packet.size);
        }

        let acked_bitrate = self.acked_bitrate_estimator.current_estimate();
        let total_acked_bitrate = self.total_acked_bitrate(acked_bitrate, now);

        // Use the latest probe result from this update, if any
        let probe_result = latest_probe_result;

        let is_probe_result = probe_result.is_some();

        // Update delay controller with the latest probe result
        let maybe_estimate =
            self.delay_controller
                .update(&acked_packets, acked_bitrate, probe_result, now);

        let Some(delay_estimate) = maybe_estimate else {
            return;
        };

        // Data channel traffic shares the path but not the transport feedback.
        let mut loss_samples: Vec<LossSample> =
            send_records.iter().map(|record| (*record).into()).collect();
        for sample in self.data_channel_samples.drain(..) {
            count += 1;
            lost += u32::from(sample.lost);
            loss_samples.push(sample);
        }

        let loss = if count == 0 {
            0.0
        } else {
            lost as f64 / count as f64
        };
        log_loss!(loss);

        // During startup with no loss, use delay-based estimate directly
        if in_startup_phase(self.started_at, now) && loss <= 0.001 {
            self.loss_controller.set_bandwidth_estimate(delay_estimate);
            return;
        }

        // When probe succeeds, set bandwidth directly
        if is_probe_result {
            self.loss_controller.set_bandwidth_estimate(delay_estimate);
        }

        // The loss controller sees data channel bytes in its sending rate, so
        // its acknowledged rate must include them too.
        if let Some(acked_bitrate) = total_acked_bitrate {
            self.loss_controller.set_acknowledged_bitrate(acked_bitrate);
        }

        // This corresponds to UpdateLossBasedEstimator + UpdateEstimate
        self.loss_controller
            .update_bandwidth_estimate(&loss_samples, delay_estimate);

        // Loss-based result is capped by delay_based_limit
        let loss_result = self.loss_controller.loss_based_result();
        if let Some(loss_estimate) = loss_result.bandwidth_estimate {
            if loss_estimate > delay_estimate {
                // Loss controller produced higher estimate than delay controller
                // Cap it at delay estimate (delay controller is the upper limit)
                self.loss_controller.set_bandwidth_estimate(delay_estimate);
            }
        }

        // Feed the (possibly combined) estimate into subcomponents wanting it.
        self.propagate_estimate();
    }

    pub fn poll_timeout(&self) -> (Option<Instant>, Reason) {
        let delay_timeout = Some(self.delay_controller.poll_timeout());
        let probe_timeout = Some(self.probe_control.poll_timeout());
        let probe_estimator_timeout = Some(self.probe_estimator.poll_timeout());
        (delay_timeout, Reason::BweDelayControl)
            .soonest((probe_timeout, Reason::BweProbeControl))
            .soonest((probe_estimator_timeout, Reason::BweProbeEstimator))
    }

    /// Handle periodic timeout for BWE components.
    pub fn handle_timeout(
        &mut self,
        desired_bitrate: Bitrate,
        do_probe: bool,
        now: Instant,
    ) -> Option<ProbeClusterConfig> {
        self.delay_controller
            .handle_timeout(self.acked_bitrate_estimator.current_estimate(), now);

        // Update probe control with desired bitrate.
        self.probe_control.set_desired_bitrate(desired_bitrate);

        // Get ALR state and forward to both probe control and loss controller
        let alr_start_time = self.alr_detector.alr_start_time();
        if let Some(t) = alr_start_time {
            self.probe_control.set_alr_start_time(t);
        } else {
            self.probe_control.set_alr_stop_time(now);
        }

        self.loss_controller.set_alr_start_time(alr_start_time);

        // Get link capacity estimate and forward to loss controller.
        let link_capacity = self.link_capacity_estimator.capacity_estimate(now);
        self.loss_controller
            .set_link_capacity_estimate(link_capacity);

        // Clean up expired probe cluster state
        self.probe_estimator.handle_timeout(now);

        // Feed the current estimate into subcontrollers, if it changed.
        self.propagate_estimate();

        // If we can't probe, clear any pending/active probes
        if !do_probe {
            self.probe_estimator.clear_probes();
        }

        self.probe_control.enable(do_probe);

        // Timer-driven probe logic (WebRTC `Process()` equivalent).
        self.probe_control.handle_timeout(now)
    }

    fn propagate_estimate(&mut self) {
        // Do we have a value?
        let Some(estimate) = self.last_estimate() else {
            return;
        };
        // Did it change?
        if self.last_updated_estimate == Some(estimate) {
            return;
        }

        let cause = self.bandwidth_limited_cause();

        self.probe_control.set_estimated_bitrate(estimate, cause);
        self.alr_detector.set_estimated_bitrate(estimate);

        // Don't update until this changes.
        self.last_updated_estimate = Some(estimate);
    }

    fn bandwidth_limited_cause(&self) -> BandwidthLimitedCause {
        if self.delay_controller.is_overusing() {
            return BandwidthLimitedCause::DelayBasedLimitedDelayIncreased;
        }

        match self.loss_controller.loss_based_result().state {
            LossControllerState::DelayBased => BandwidthLimitedCause::DelayBasedLimited,
            LossControllerState::Increasing => BandwidthLimitedCause::LossLimitedBweIncreasing,
            LossControllerState::Decreasing => BandwidthLimitedCause::LossLimitedBwe,
        }
    }

    /// Get the latest estimate.
    pub fn last_estimate(&self) -> Option<Bitrate> {
        let delay_estimate = self.delay_controller.last_estimate();

        let loss_result = self.loss_controller.loss_based_result();

        // Only apply loss-based limiting when actively in a loss-limiting state
        match loss_result.state {
            LossControllerState::DelayBased => {
                // Loss controller defers to delay-based estimate
                delay_estimate
            }
            LossControllerState::Decreasing | LossControllerState::Increasing => {
                // Loss controller is actively limiting or recovering
                match (delay_estimate, loss_result.bandwidth_estimate) {
                    (Some(de), Some(le)) => Some(de.min(le)),
                    (None, le @ Some(_)) => le,
                    (de @ Some(_), None) => de,
                    (None, None) => None,
                }
            }
        }
    }

    /// Start analyzing a probe cluster.
    ///
    /// This should be called when the pacer starts sending a probe cluster,
    /// to tell the estimator which cluster to watch for in TWCC feedback.
    /// Returns `true` if the probe was started, `false` if rejected.
    pub fn start_probe(&mut self, config: ProbeClusterConfig, now: Instant) -> bool {
        self.probe_estimator.probe_start(config, now)
    }

    /// End a probe cluster and mark it for cleanup.
    ///
    /// This should be called when the pacer finishes sending a probe cluster.
    /// The estimator will continue collecting feedback for a cluster history period
    /// to allow late-arriving TWCC reports to refine the estimate.
    pub fn end_probe(&mut self, now: Instant, cluster_id: TwccClusterId) {
        self.probe_estimator.end_probe(now, cluster_id);
    }

    pub fn reset(&mut self, init_bitrate: Bitrate) {
        *self = Self::new(init_bitrate);
    }
}

/// A RTP packet that has been sent and acknowledged by the receiver in a TWCC report.
#[derive(Debug, Copy, Clone)]
pub struct AckedPacket {
    /// The TWCC sequence number
    seq_no: TwccSeq,
    /// The size of the packets in bytes.
    size: DataSize,
    /// When we sent the packet
    local_send_time: Instant,
    /// When the packet was received at the remote, note this Instant is only usable with other
    /// instants of the same type i.e. those that represent a TWCC reported receive time for this
    /// session.
    remote_recv_time: Instant,
    /// The local time when received confirmation that the other side received the seq i.e. when we
    /// received the TWCC report for this packet.
    local_recv_time: Instant,
}

impl AckedPacket {
    fn rtt(&self) -> Duration {
        self.local_recv_time - self.local_send_time
    }

    fn order_by_receive_time(lhs: &Self, rhs: &Self) -> Ordering {
        if lhs.remote_recv_time != rhs.remote_recv_time {
            lhs.remote_recv_time.cmp(&rhs.remote_recv_time)
        } else if lhs.local_send_time != rhs.local_send_time {
            lhs.local_send_time.cmp(&rhs.local_send_time)
        } else {
            lhs.seq_no.cmp(&rhs.seq_no)
        }
    }
}

// NB: Extracted for lifetime reasons
fn in_startup_phase(started_at: Option<Instant>, now: Instant) -> bool {
    started_at
        .map(|s| now.duration_since(s) <= STARTUP_PHASE)
        .unwrap_or(false)
}

impl TryFrom<&TwccSendRecord> for AckedPacket {
    type Error = ();

    fn try_from(value: &TwccSendRecord) -> Result<Self, Self::Error> {
        let Some(remote_recv_time) = value.remote_recv_time() else {
            return Err(());
        };
        let Some(local_recv_time) = value.local_recv_time() else {
            return Err(());
        };

        Ok(Self {
            seq_no: value.seq(),
            size: value.size().into(),
            local_send_time: value.local_send_time(),
            remote_recv_time,
            local_recv_time,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandwidthUsage {
    Overuse,
    Normal,
    Underuse,
}

impl fmt::Display for BandwidthUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BandwidthUsage::Overuse => write!(f, "overuse"),
            BandwidthUsage::Normal => write!(f, "normal"),
            BandwidthUsage::Underuse => write!(f, "underuse"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::rtp_::TwccPacketId;

    const ONE_WAY: Duration = Duration::from_millis(20);
    const FEEDBACK_INTERVAL: Duration = Duration::from_millis(100);
    const PACKET: usize = 1000;

    /// Drives the estimator with paced RTP that always arrives beside data
    /// channel chunks, some of which need a retransmission. The senders do
    /// not follow the estimate, like a policer dropping bursts of a data
    /// channel that ignores it.
    struct Sim {
        bwe: SendSideBandwidthEstimator,
        now: Instant,
        seq: u64,
        rtp: Bitrate,
        data_channel: Bitrate,
        /// One lost transmission every this many delivered chunks.
        loss_every: Option<u64>,
        chunks: u64,
        feed_data_channel: bool,
    }

    impl Sim {
        fn new(feed_data_channel: bool, loss_every: Option<u64>) -> Self {
            Sim {
                bwe: SendSideBandwidthEstimator::new(Bitrate::kbps(2500)),
                now: Instant::now(),
                seq: 0,
                rtp: Bitrate::kbps(600),
                data_channel: Bitrate::kbps(1000),
                loss_every,
                chunks: 0,
                feed_data_channel,
            }
        }

        fn interval(rate: Bitrate) -> Duration {
            Duration::from_secs_f64(PACKET as f64 * 8.0 / rate.as_f64())
        }

        fn run(&mut self, duration: Duration) {
            let end = self.now + duration;
            while self.now < end {
                let start = self.now;
                self.now += FEEDBACK_INTERVAL;

                let mut records = vec![];
                let mut sent = start;
                while sent < self.now {
                    records.push(TwccSendRecord::test_new(
                        TwccPacketId::new(self.seq),
                        sent,
                        PACKET,
                        sent + 2 * ONE_WAY,
                        Some(sent + ONE_WAY),
                    ));
                    self.seq += 1;
                    sent += Self::interval(self.rtp);
                }

                let mut sent = start;
                while self.feed_data_channel && sent < self.now {
                    self.chunks += 1;
                    let lost = self.loss_every.is_some_and(|n| self.chunks % n == 0);
                    let retransmitted = sent + Duration::from_millis(50);
                    let last_sent = if lost { retransmitted } else { sent };
                    self.bwe.on_data_channel_delivery(DataChannelDelivery {
                        first_sent: sent,
                        last_sent,
                        acked_at: last_sent + 2 * ONE_WAY,
                        size: PACKET.into(),
                        lost_transmissions: u32::from(lost),
                    });
                    sent += Self::interval(self.data_channel);
                }

                self.bwe.update(records.iter(), self.now);
                self.bwe.handle_timeout(Bitrate::mbps(5), false, self.now);
            }
        }

        fn state(&self) -> LossControllerState {
            self.bwe.loss_controller.loss_based_result().state
        }

        fn estimate(&self) -> Bitrate {
            self.bwe.last_estimate().expect("an estimate")
        }
    }

    #[test]
    fn data_channel_loss_reaches_the_loss_controller() {
        // Every other chunk loses its first transmission: a third of the
        // data channel transmissions and a quarter of all bytes sent.
        let mut blind = Sim::new(false, Some(2));
        blind.run(Duration::from_secs(8));
        assert_eq!(blind.state(), LossControllerState::DelayBased);
        assert!(
            blind.estimate() >= Bitrate::kbps(2400),
            "RTP alone shows no loss: {}",
            blind.estimate()
        );

        let mut fed = Sim::new(true, Some(2));
        fed.run(Duration::from_secs(8));
        assert_eq!(fed.state(), LossControllerState::Decreasing);
        // 1.6 Mbps leaves the sender while a quarter of it is dropped, so the
        // estimate falls below the sending rate.
        assert!(
            fed.estimate() < Bitrate::kbps(1600),
            "data channel loss must limit the estimate: {}",
            fed.estimate()
        );
    }

    #[test]
    fn lossless_data_channel_traffic_keeps_the_estimate() {
        let mut fed = Sim::new(true, None);
        fed.run(Duration::from_secs(8));
        assert_eq!(fed.state(), LossControllerState::DelayBased);
        assert!(fed.estimate() >= Bitrate::kbps(2400), "{}", fed.estimate());
    }

    #[test]
    fn data_channel_acknowledgements_count_toward_sustained_throughput() {
        let mut fed = Sim::new(true, None);
        fed.run(Duration::from_secs(3));
        let now = fed.now;
        let data_channel = fed.bwe.data_channel_acked_bitrate(now);
        assert!(
            data_channel >= Bitrate::kbps(900) && data_channel <= Bitrate::kbps(1100),
            "{data_channel}"
        );
        let total = fed
            .bwe
            .total_acked_bitrate(Some(Bitrate::kbps(600)), now)
            .unwrap();
        assert_eq!(total, Bitrate::kbps(600) + data_channel);

        // The loss controller's sending rate includes data channel bytes, so
        // its acknowledged rate must as well.
        let seen = fed.bwe.loss_controller.acknowledged_bitrate();
        assert!(seen >= Bitrate::kbps(1450), "{seen}");

        // Acknowledgements leave the window once traffic stops.
        assert_eq!(
            fed.bwe
                .data_channel_acked_bitrate(now + DATA_CHANNEL_ACKED_WINDOW * 2),
            Bitrate::ZERO
        );
    }
}
