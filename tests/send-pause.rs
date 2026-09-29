//! Pausing a sending m-line with `DirectApi::pause_send`.
//!
//! Every test captures the sender's outgoing RTP (`RawPacket::RtpTx`) and asserts on the
//! video and RTX SSRCs. Stopping writes alone does not quiet a sending m-line: the pacer
//! keeps padding it, and padding can resend cached video. The pause must stop all of it.

use std::net::Ipv4Addr;
use std::time::{Duration, Instant};

use netem::{NetemConfig, Probability, RandomLoss};
use str0m::bwe::Bitrate;
use str0m::media::{Direction, MediaKind, MediaTime, Mid, Pt};
use str0m::rtp::rtcp::Rtcp;
use str0m::rtp::{RawPacket, Ssrc};
use str0m::{Event, RtcError};

mod common;
use common::{Peer, TestRtc, init_crypto_default, init_log, negotiate, progress};

const FRAME_INTERVAL: Duration = Duration::from_millis(33);

/// Marks the samples written before the pause.
const MARK_BEFORE: u8 = 0x3c;
/// Marks a sample written, but not packetized, right before the pause.
const MARK_DROPPED: u8 = 0x77;
/// Marks the keyframe written after the resume.
const MARK_RESUMED: u8 = 0xa5;

struct Pair {
    l: TestRtc,
    r: TestRtc,
    mid: Mid,
    pt: Pt,
    ssrc: Ssrc,
    rtx: Ssrc,
    frames: u64,
}

/// A VP8 sample: a three byte frame tag (P bit clear for a keyframe) and a marker body.
fn vp8_sample(size: usize, keyframe: bool, marker: u8) -> Vec<u8> {
    let mut data = vec![marker; size];
    // Frame tag: bit 0 is the inverse keyframe flag, bit 4 is show_frame.
    data[0] = if keyframe { 0x10 } else { 0x11 };
    data[1] = 0x00;
    data[2] = 0x00;
    data
}

fn connect(initial: Bitrate, desired: Bitrate) -> Result<Pair, RtcError> {
    let mut l = TestRtc::new_with_config(Peer::Left, |c| {
        c.enable_bwe(Some(initial)).enable_raw_packets(true)
    });
    let mut r = TestRtc::new(Peer::Right);

    l.add_host_candidate((Ipv4Addr::new(1, 1, 1, 1), 1000).into());
    r.add_host_candidate((Ipv4Addr::new(2, 2, 2, 2), 2000).into());

    let mid = negotiate(&mut l, &mut r, |change| {
        change.add_media(MediaKind::Video, Direction::SendOnly, None, None, None)
    });

    loop {
        if l.is_connected() && r.is_connected() {
            break;
        }
        progress(&mut l, &mut r)?;
    }

    let max = l.last.max(r.last);
    l.last = max;
    r.last = max;

    l.bwe().set_desired_bitrate(desired);

    let pt = l.params_vp8().pt();
    let (ssrc, rtx) = {
        let mut api = l.direct_api();
        let stream = api.stream_tx_by_mid(mid, None).expect("send stream");
        (stream.ssrc(), stream.rtx().expect("RTX for VP8"))
    };

    Ok(Pair {
        l,
        r,
        mid,
        pt,
        ssrc,
        rtx,
        frames: 0,
    })
}

impl Pair {
    fn write(&mut self, size: usize, keyframe: bool, marker: u8) -> Result<(), RtcError> {
        let wallclock = self.l.last;
        let rtp_time = MediaTime::from_90khz(self.frames * 3000);
        self.frames += 1;
        let pt = self.pt;
        self.l.writer(self.mid).expect("writer").write(
            pt,
            wallclock,
            rtp_time,
            vp8_sample(size, keyframe, marker),
        )
    }

    /// Runs for `duration`, writing one sample of `frame_size` every frame interval when set.
    fn run(&mut self, duration: Duration, frame_size: Option<usize>) -> Result<(), RtcError> {
        let end = self.l.last + duration;
        let mut next_frame = self.l.last;
        while self.l.last < end {
            if let Some(size) = frame_size {
                if self.l.last >= next_frame {
                    self.write(size, self.frames == 0, MARK_BEFORE)?;
                    next_frame += FRAME_INTERVAL;
                }
            }
            progress(&mut self.l, &mut self.r)?;
        }
        Ok(())
    }

    /// Runs until an event matching `found` appears after `from`, or `limit` passes.
    fn run_until(
        &mut self,
        from: usize,
        limit: Duration,
        frame_size: Option<usize>,
        found: impl Fn(&Event) -> bool,
    ) -> Result<bool, RtcError> {
        let end = self.l.last + limit;
        let mut next_frame = self.l.last;
        while self.l.last < end {
            if self.l.events[from..].iter().any(|(_, e)| found(e)) {
                return Ok(true);
            }
            if let Some(size) = frame_size {
                if self.l.last >= next_frame {
                    self.write(size, self.frames == 0, MARK_BEFORE)?;
                    next_frame += FRAME_INTERVAL;
                }
            }
            progress(&mut self.l, &mut self.r)?;
        }
        Ok(self.l.events[from..].iter().any(|(_, e)| found(e)))
    }

    fn mark(&self) -> usize {
        self.l.events.len()
    }

    /// Outgoing RTP on the video and RTX SSRCs after event index `from`.
    fn video_rtp_since(&self, from: usize) -> Vec<(Instant, Ssrc, u16, Vec<u8>)> {
        self.l.events[from..]
            .iter()
            .filter_map(|(at, e)| match e.as_raw_packet() {
                Some(RawPacket::RtpTx(header, buf))
                    if header.ssrc == self.ssrc || header.ssrc == self.rtx =>
                {
                    Some((
                        *at,
                        header.ssrc,
                        header.sequence_number,
                        buf[header.header_len..].to_vec(),
                    ))
                }
                _ => None,
            })
            .collect()
    }

    /// Outgoing RTP on any SSRC after event index `from`.
    fn any_rtp_since(&self, from: usize) -> usize {
        self.l.events[from..]
            .iter()
            .filter(|(_, e)| matches!(e.as_raw_packet(), Some(RawPacket::RtpTx(..))))
            .count()
    }

    fn rtcp_tx_since(&self, from: usize) -> usize {
        self.l.events[from..]
            .iter()
            .filter(|(_, e)| matches!(e.as_raw_packet(), Some(RawPacket::RtcpTx(_))))
            .count()
    }

    fn nacks_rx_since(&self, from: usize) -> usize {
        self.l.events[from..]
            .iter()
            .filter(|(_, e)| matches!(e.as_raw_packet(), Some(RawPacket::RtcpRx(Rtcp::Nack(_)))))
            .count()
    }

    fn queued_packets(&mut self) -> usize {
        let mut api = self.l.direct_api();
        let stream = api.stream_tx_by_mid(self.mid, None).expect("send stream");
        stream.queue_info().map(|q| q.packet_count()).unwrap_or(0)
    }
}

fn has_run(body: &[u8], marker: u8) -> bool {
    body.windows(16).any(|w| w.iter().all(|b| *b == marker))
}

fn setup_streaming() -> Result<Pair, RtcError> {
    init_log();
    init_crypto_default();

    let mut pair = connect(Bitrate::kbps(300), Bitrate::kbps(1_500))?;
    pair.l.set_forced_time_advance(Duration::from_millis(1));
    pair.r.set_forced_time_advance(Duration::from_millis(1));
    pair.run(Duration::from_secs(3), Some(2_000))?;
    Ok(pair)
}

/// Without a pause, a sending m-line with no new samples keeps emitting RTP: padding and
/// spurious resends of cached video on the RTX SSRC. This is what the gate must stop.
#[test]
fn control_stopping_writes_leaves_padding_on_the_video_line() -> Result<(), RtcError> {
    let mut pair = setup_streaming()?;

    let stopped = pair.mark();
    pair.run(Duration::from_secs(3), None)?;

    let after = pair.video_rtp_since(stopped);
    let rtx = after.iter().filter(|p| p.1 == pair.rtx).count();
    assert!(
        rtx > 20,
        "padding keeps the RTX SSRC busy after writes stop, saw {rtx} packets"
    );
    let resent_video = after
        .iter()
        .filter(|p| p.1 == pair.rtx && has_run(&p.3, MARK_BEFORE))
        .count();
    assert!(
        resent_video > 0,
        "padding resends cached video after writes stop"
    );

    Ok(())
}

/// Pausing with a full send queue and RTX cache, then receiving NACKs that were in flight,
/// and no further TWCC feedback: nothing on the video or RTX SSRCs, while RTCP continues.
#[test]
fn pause_with_full_queues_and_delayed_nacks_sends_nothing() -> Result<(), RtcError> {
    let mut pair = setup_streaming()?;

    // Lose some video on the way to R so it NACKs, and delay R's feedback so NACKs and TWCC
    // reports generated before the pause arrive after it.
    pair.r.set_netem(
        NetemConfig::new()
            .loss(RandomLoss::new(Probability::new(0.1)))
            .seed(7),
    );
    pair.l
        .set_netem(NetemConfig::new().latency(Duration::from_millis(400)));
    pair.run(Duration::from_secs(2), Some(2_000))?;

    // A large sample fills the send queue beyond what the pacer releases at once.
    pair.write(60_000, false, MARK_BEFORE)?;
    progress(&mut pair.l, &mut pair.r)?;
    let queued = pair.queued_packets();
    assert!(queued > 0, "the send queue holds packets at the pause");
    let before = pair.video_rtp_since(0).len();
    assert!(
        before > 100,
        "the RTX cache holds sent video, {before} packets"
    );

    let paused = pair.mark();
    assert!(pair.l.direct_api().pause_send(pair.mid));
    assert!(pair.l.media(pair.mid).unwrap().is_send_paused());

    // Deliver the delayed feedback, then withhold further feedback entirely.
    pair.run(Duration::from_millis(800), None)?;
    let nacks = pair.nacks_rx_since(paused);
    assert!(nacks > 0, "NACKs for cached video arrived after the pause");
    pair.l
        .set_netem(NetemConfig::new().loss(RandomLoss::new(Probability::new(1.0))));
    pair.run(Duration::from_secs(4), None)?;

    let leaked = pair.video_rtp_since(paused);
    assert!(
        leaked.is_empty(),
        "no RTP on the video or RTX SSRCs after the pause, saw {}: {:?}",
        leaked.len(),
        leaked
            .iter()
            .take(5)
            .map(|p| (p.1, p.2))
            .collect::<Vec<_>>()
    );
    assert_eq!(pair.any_rtp_since(paused), 0, "no RTP on any SSRC");
    assert!(
        pair.rtcp_tx_since(paused) > 0,
        "RTCP continues while the send is paused"
    );

    Ok(())
}

/// Pausing while a bandwidth probe cluster is in progress stops its padding.
#[test]
fn pause_during_an_active_probe_sends_nothing() -> Result<(), RtcError> {
    init_log();
    init_crypto_default();

    let mut pair = connect(Bitrate::kbps(300), Bitrate::mbps(5))?;
    pair.l.set_forced_time_advance(Duration::from_micros(250));
    pair.r.set_forced_time_advance(Duration::from_micros(250));
    pair.run(Duration::from_millis(500), Some(1_000))?;

    // Raise the target so the estimator probes for it, then pause as the cluster starts.
    let from = pair.mark();
    pair.l.bwe().set_desired_bitrate(Bitrate::mbps(20));
    let probing = pair.run_until(from, Duration::from_secs(20), Some(1_000), |e| {
        matches!(e, Event::Probe(_))
    })?;
    assert!(probing, "a probe cluster started");

    let paused = pair.mark();
    assert!(pair.l.direct_api().pause_send(pair.mid));
    pair.run(Duration::from_secs(3), None)?;

    let leaked = pair.video_rtp_since(paused);
    assert!(
        leaked.is_empty(),
        "no probe padding after the pause, saw {} packets",
        leaked.len()
    );
    assert_eq!(pair.any_rtp_since(paused), 0, "no RTP on any SSRC");

    Ok(())
}

/// A sample written but not yet packetized is dropped by the pause and never sent.
#[test]
fn pause_right_after_an_unpacketized_write_drops_it() -> Result<(), RtcError> {
    let mut pair = setup_streaming()?;

    pair.write(3_000, false, MARK_DROPPED)?;
    let paused = pair.mark();
    assert!(pair.l.direct_api().pause_send(pair.mid));
    pair.run(Duration::from_secs(3), None)?;

    let leaked = pair.video_rtp_since(paused);
    assert!(
        leaked.is_empty(),
        "no RTP on the video or RTX SSRCs after the pause, saw {}",
        leaked.len()
    );
    let everything = pair.video_rtp_since(0);
    assert!(
        !everything.iter().any(|p| has_run(&p.3, MARK_DROPPED)),
        "the dropped sample never left"
    );

    Ok(())
}

/// Writing on a paused m-line is refused and queues nothing.
#[test]
fn write_while_paused_is_refused() -> Result<(), RtcError> {
    let mut pair = setup_streaming()?;

    assert!(pair.l.direct_api().pause_send(pair.mid));
    let paused = pair.mark();
    let refused = pair.write(3_000, true, MARK_DROPPED);
    assert!(
        matches!(refused, Err(RtcError::SendPaused(mid)) if mid == pair.mid),
        "write on a paused m-line fails with SendPaused, got {refused:?}"
    );
    pair.run(Duration::from_secs(1), None)?;
    assert!(pair.video_rtp_since(paused).is_empty());

    // An unknown mid is reported.
    assert!(!pair.l.direct_api().pause_send("nope".into()));
    assert!(!pair.l.direct_api().resume_send("nope".into()));

    Ok(())
}

/// After a resume, the first packet on the video or RTX SSRCs carries the keyframe written
/// after it; no padding precedes it, and padding may return afterwards.
#[test]
fn resume_sends_the_new_keyframe_first() -> Result<(), RtcError> {
    let mut pair = setup_streaming()?;

    let before = pair.video_rtp_since(0);
    let last_seq = before
        .iter()
        .filter(|p| p.1 == pair.ssrc)
        .map(|p| p.2)
        .next_back()
        .expect("video was sent");

    assert!(pair.l.direct_api().pause_send(pair.mid));
    pair.run(Duration::from_secs(2), None)?;

    let resumed = pair.mark();
    assert!(pair.l.direct_api().resume_send(pair.mid));
    assert!(!pair.l.media(pair.mid).unwrap().is_send_paused());

    // Let the pacer run without a sample for a while: still nothing, padding included.
    pair.run(Duration::from_millis(500), None)?;
    assert!(
        pair.video_rtp_since(resumed).is_empty(),
        "no padding before the first sample after the resume"
    );

    pair.write(3_000, true, MARK_RESUMED)?;
    pair.run(Duration::from_secs(3), None)?;

    let after = pair.video_rtp_since(resumed);
    let first = after.first().expect("the keyframe was sent");
    assert_eq!(first.1, pair.ssrc, "the first packet is media, not RTX");
    assert!(
        has_run(&first.3, MARK_RESUMED),
        "it carries the new keyframe"
    );
    assert_eq!(
        first.2,
        last_seq.wrapping_add(1),
        "the video sequence continues from the last packet before the pause"
    );
    let rtx = after.iter().filter(|p| p.1 == pair.rtx).count();
    assert!(rtx > 0, "padding returns once the keyframe has been sent");

    Ok(())
}

/// With the paused m-line the only sending media, padding stops at once.
#[test]
fn pause_stops_padding_immediately() -> Result<(), RtcError> {
    let mut pair = setup_streaming()?;

    // Stop writing and wait until the line is only padding.
    let idle = pair.mark();
    let rtx = pair.rtx;
    let padding = pair.run_until(
        idle,
        Duration::from_secs(3),
        None,
        |e| matches!(e.as_raw_packet(), Some(RawPacket::RtpTx(h, _)) if h.ssrc == rtx),
    );
    assert!(padding?, "the idle line pads");

    let paused = pair.mark();
    assert!(pair.l.direct_api().pause_send(pair.mid));
    pair.run(Duration::from_millis(100), None)?;
    assert_eq!(pair.any_rtp_since(paused), 0, "no RTP within 100 ms");
    pair.run(Duration::from_secs(5), None)?;
    assert_eq!(pair.any_rtp_since(paused), 0, "no RTP within 5 s");

    Ok(())
}

/// The pause belongs to the media, so an ICE restart renegotiated by the remote keeps it.
#[test]
fn pause_survives_a_remote_ice_restart() -> Result<(), RtcError> {
    let mut pair = setup_streaming()?;

    let paused = pair.mark();
    assert!(pair.l.direct_api().pause_send(pair.mid));
    negotiate(&mut pair.r, &mut pair.l, |change| {
        change.ice_restart(true);
    });
    pair.run(Duration::from_secs(3), None)?;

    assert!(pair.l.media(pair.mid).unwrap().is_send_paused());
    assert!(
        pair.video_rtp_since(paused).is_empty(),
        "no RTP on the video or RTX SSRCs across the restart"
    );

    let resumed = pair.mark();
    assert!(pair.l.direct_api().resume_send(pair.mid));
    pair.write(3_000, true, MARK_RESUMED)?;
    pair.run(Duration::from_secs(1), None)?;
    let after = pair.video_rtp_since(resumed);
    assert!(
        after
            .first()
            .is_some_and(|p| p.1 == pair.ssrc && has_run(&p.3, MARK_RESUMED)),
        "the keyframe leads after the restart"
    );

    Ok(())
}
