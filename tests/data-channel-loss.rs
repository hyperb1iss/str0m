//! Request/response latency over a lossy data channel.

use std::net::Ipv4Addr;
use std::time::{Duration, Instant};

use netem::{NetemConfig, Probability, RandomLoss};
use str0m::channel::ChannelId;
use str0m::{Event, RtcError};

mod common;
use common::{Peer, TestRtc, init_crypto_default, init_log, progress};

const ONE_WAY: Duration = Duration::from_millis(25);
const LOSS: f32 = 0.05;
const EXCHANGES: usize = 400;
const THINK_TIME: Duration = Duration::from_millis(20);
/// Poll each peer at least this often, so writes leave and events are
/// stamped within a tick of when they happen.
const TICK: Duration = Duration::from_millis(1);

struct Exchange {
    started: Instant,
    response_len: usize,
}

fn open_channel(l: &mut TestRtc, r: &mut TestRtc) -> Result<(ChannelId, ChannelId), RtcError> {
    l.add_host_candidate((Ipv4Addr::new(1, 1, 1, 1), 1000).into());
    r.add_host_candidate((Ipv4Addr::new(2, 2, 2, 2), 2000).into());

    let mut change = l.sdp_api();
    let client = change.add_channel("rpc".into());
    let (offer, pending) = change.apply().unwrap();
    let answer = r.rtc.sdp_api().accept_offer(offer)?;
    l.rtc.sdp_api().accept_answer(pending, answer)?;

    loop {
        progress(l, r)?;
        let client_open = l
            .events
            .iter()
            .any(|(_, e)| matches!(e, Event::ChannelOpen(id, _) if *id == client));
        let server = r.events.iter().find_map(|(_, e)| match e {
            Event::ChannelOpen(id, _) => Some(*id),
            _ => None,
        });
        if let (true, Some(server)) = (client_open, server) {
            return Ok((client, server));
        }
        assert!(l.duration() < Duration::from_secs(10), "channel must open");
    }
}

fn percentile(sorted: &[Duration], percent: usize) -> Duration {
    sorted[(sorted.len() * percent).div_ceil(100) - 1]
}

/// Sequential API calls: a small request, then a response of either a
/// single chunk or several fragments. Each call measures the client's time
/// from writing the request to reading the whole response.
fn exchange_completion_times() -> Result<Vec<Duration>, RtcError> {
    let mut l = TestRtc::new(Peer::Left);
    let mut r = TestRtc::new(Peer::Right);
    let (client, server) = open_channel(&mut l, &mut r)?;
    l.set_forced_time_advance(TICK);
    r.set_forced_time_advance(TICK);

    let lossy = |seed| {
        NetemConfig::new()
            .latency(ONE_WAY)
            .loss(RandomLoss::new(Probability::new(LOSS)))
            .seed(seed)
    };
    l.set_netem(lossy(7));
    r.set_netem(lossy(11));

    let mut seen = [l.events.len(), r.events.len()];
    let mut current: Option<Exchange> = None;
    let mut next_start = l.last + THINK_TIME;
    let mut completions = Vec::with_capacity(EXCHANGES);
    while completions.len() < EXCHANGES {
        if current.is_none() && l.last >= next_start {
            let index = completions.len();
            let response_len = if index % 2 == 0 { 300 } else { 3000 };
            let mut request = vec![0; 120];
            request[..8].copy_from_slice(&(response_len as u64).to_be_bytes());
            l.channel(client).unwrap().write(true, &request)?;
            current = Some(Exchange {
                started: l.last,
                response_len,
            });
        }

        progress(&mut l, &mut r)?;

        for (_, event) in r.events.drain(seen[1]..).collect::<Vec<_>>() {
            if let Event::ChannelData(data) = event {
                let len = u64::from_be_bytes(data.data[..8].try_into().unwrap()) as usize;
                r.channel(server).unwrap().write(true, &vec![1; len])?;
            }
        }
        seen[1] = r.events.len();

        for (at, event) in l.events.drain(seen[0]..).collect::<Vec<_>>() {
            if let Event::ChannelData(data) = event {
                let exchange = current.take().expect("a response follows a request");
                assert_eq!(data.data.len(), exchange.response_len);
                completions.push(at - exchange.started);
                next_start = at + THINK_TIME;
            }
        }
        seen[0] = l.events.len();

        assert!(
            l.duration() < Duration::from_secs(600),
            "{} of {EXCHANGES} exchanges completed",
            completions.len()
        );
    }
    Ok(completions)
}

#[test]
fn data_channel_requests_recover_from_random_loss_quickly() -> Result<(), RtcError> {
    init_log();
    init_crypto_default();

    let mut times = exchange_completion_times()?;
    times.sort();
    let rtt = 2 * ONE_WAY;
    let p50 = percentile(&times, 50);
    let p95 = percentile(&times, 95);
    let p99 = percentile(&times, 99);
    let max = *times.last().unwrap();
    let mean = times.iter().sum::<Duration>() / times.len() as u32;
    println!(
        "{EXCHANGES} exchanges, {}% loss each way, {rtt:?} RTT: \
         p50={p50:?} p95={p95:?} p99={p99:?} max={max:?} mean={mean:?}",
        LOSS * 100.0
    );

    // A clean exchange takes one RTT, plus the harness's 5 ms batching
    // window before the server replies and polling ticks. A lost tail costs
    // one probe timeout of two SRTT plus another RTT, far inside the 400 ms
    // minimum RTO.
    assert!(
        p50 <= rtt + Duration::from_millis(5) + 4 * TICK,
        "p50 {p50:?}"
    );
    assert!(p95 <= 4 * rtt, "p95 {p95:?} must avoid T3-rtx");
    assert!(max < Duration::from_millis(600), "max {max:?}");
    Ok(())
}
