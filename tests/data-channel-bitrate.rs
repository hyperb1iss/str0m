use std::net::Ipv4Addr;
use std::time::Duration;

use str0m::bwe::Bitrate;
use str0m::{Event, RtcError};

mod common;
use common::{Peer, TestRtc, init_crypto_default, init_log, progress};

/// The data channel send rate counts the DTLS datagrams carrying SCTP, so an
/// application can take it out of the media budget, and it falls back to zero
/// once the channel goes quiet.
#[test]
pub fn data_channel_bitrate_tracks_the_send_rate() -> Result<(), RtcError> {
    init_log();
    init_crypto_default();

    let mut l = TestRtc::new(Peer::Left);
    let mut r = TestRtc::new(Peer::Right);
    l.add_host_candidate((Ipv4Addr::new(1, 1, 1, 1), 1000).into());
    r.add_host_candidate((Ipv4Addr::new(2, 2, 2, 2), 2000).into());

    let mut change = l.sdp_api();
    let cid = change.add_channel("bulk".into());
    let (offer, pending) = change.apply().unwrap();
    let answer = r.rtc.sdp_api().accept_offer(offer)?;
    l.rtc.sdp_api().accept_answer(pending, answer)?;

    loop {
        progress(&mut l, &mut r)?;
        let open = l
            .events
            .iter()
            .any(|(_, e)| matches!(e, Event::ChannelOpen(id, _) if *id == cid));
        if open {
            break;
        }
        assert!(l.duration() < Duration::from_secs(10), "channel must open");
    }
    l.set_forced_time_advance(Duration::from_millis(1));
    r.set_forced_time_advance(Duration::from_millis(1));

    // 1000 bytes every 10 ms is 800 kbps of messages.
    let message = vec![9; 1000];
    let interval = Duration::from_millis(10);
    let start = l.duration();
    let mut next = start;
    while l.duration() < start + Duration::from_secs(3) {
        if l.duration() >= next {
            l.channel(cid).unwrap().write(true, &message)?;
            next += interval;
        }
        progress(&mut l, &mut r)?;
    }

    // SCTP and DTLS framing add about 60 bytes to each message.
    let sending = l.bwe().data_channel_bitrate();
    assert!(
        sending >= Bitrate::kbps(800) && sending <= Bitrate::kbps(900),
        "{sending}"
    );
    // The receiver only returns acknowledgements.
    let acknowledging = r.bwe().data_channel_bitrate();
    assert!(acknowledging < Bitrate::kbps(100), "{acknowledging}");

    let quiet = l.duration();
    while l.duration() < quiet + Duration::from_millis(1500) {
        progress(&mut l, &mut r)?;
    }
    assert_eq!(l.bwe().data_channel_bitrate(), Bitrate::ZERO);

    Ok(())
}
