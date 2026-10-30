use std::time::Instant;

use str0m::Rtc;
use str0m::change::SdpOffer;
use str0m::media::{Direction, MediaKind};

mod common;

const CAPABILITY: &str = "a=rtcp-xr:rcvr-rtt=sender";

fn assert_capabilities_per_section(sdp: &str) {
    for section in sdp.split("\r\nm=").skip(1) {
        let rtp = section.starts_with("audio ") || section.starts_with("video ");
        let sending = section
            .lines()
            .any(|line| line == "a=sendonly" || line == "a=sendrecv");
        assert_eq!(section.contains(CAPABILITY), rtp && sending, "{section}");
    }
}

#[test]
fn receiver_rtt_capability_follows_sender_direction() {
    common::init_crypto_default();

    for kind in [MediaKind::Audio, MediaKind::Video] {
        for direction in [
            Direction::SendOnly,
            Direction::RecvOnly,
            Direction::SendRecv,
            Direction::Inactive,
        ] {
            let now = Instant::now();
            let mut offerer = Rtc::new(now);
            let mut answerer = Rtc::new(now);
            let mut change = offerer.sdp_api();
            change.add_media(kind, direction, None, None, None);
            let (offer, pending) = change.apply().unwrap();
            let offer_sdp = offer.to_sdp_string();
            assert_eq!(offer_sdp.contains(CAPABILITY), direction.is_sending());

            let offer = SdpOffer::from_sdp_string(&offer_sdp).unwrap();
            assert_eq!(offer.to_sdp_string(), offer_sdp);
            let answer = answerer.sdp_api().accept_offer(offer).unwrap();
            assert_eq!(
                answer.to_sdp_string().contains(CAPABILITY),
                direction.is_receiving()
            );
            offerer.sdp_api().accept_answer(pending, answer).unwrap();
        }
    }
}

#[test]
fn receiver_rtt_capability_tracks_renegotiation() {
    common::init_crypto_default();
    let now = Instant::now();
    let mut offerer = Rtc::new(now);
    let mut answerer = Rtc::new(now);
    let mut change = offerer.sdp_api();
    let mid = change.add_media(MediaKind::Video, Direction::SendOnly, None, None, None);
    let (offer, pending) = change.apply().unwrap();
    let answer = answerer.sdp_api().accept_offer(offer).unwrap();
    offerer.sdp_api().accept_answer(pending, answer).unwrap();

    for direction in [
        Direction::RecvOnly,
        Direction::Inactive,
        Direction::SendRecv,
        Direction::SendOnly,
    ] {
        let mut change = offerer.sdp_api();
        change.set_direction(mid, direction);
        let (offer, pending) = change.apply().unwrap();
        assert_eq!(
            offer.to_sdp_string().contains(CAPABILITY),
            direction.is_sending()
        );
        let answer = answerer.sdp_api().accept_offer(offer).unwrap();
        assert_capabilities_per_section(&answer.to_sdp_string());
        offerer.sdp_api().accept_answer(pending, answer).unwrap();
    }
}

#[test]
fn receiver_rtt_capability_is_scoped_to_each_media_section() {
    common::init_crypto_default();
    let now = Instant::now();
    let mut offerer = Rtc::new(now);
    let mut answerer = Rtc::new(now);
    let mut change = offerer.sdp_api();
    change.add_media(MediaKind::Video, Direction::SendOnly, None, None, None);
    change.add_media(MediaKind::Audio, Direction::RecvOnly, None, None, None);
    change.add_media(MediaKind::Video, Direction::Inactive, None, None, None);
    change.add_channel("control".into());
    let (offer, pending) = change.apply().unwrap();
    assert_capabilities_per_section(&offer.to_sdp_string());
    let answer = answerer.sdp_api().accept_offer(offer).unwrap();
    assert_capabilities_per_section(&answer.to_sdp_string());
    offerer.sdp_api().accept_answer(pending, answer).unwrap();
}

#[test]
fn unknown_xr_capabilities_do_not_expand_local_response_promise() {
    common::init_crypto_default();
    let now = Instant::now();
    let mut offerer = Rtc::new(now);
    let mut answerer = Rtc::new(now);
    let mut change = offerer.sdp_api();
    change.add_media(MediaKind::Video, Direction::RecvOnly, None, None, None);
    change.add_channel("control".into());
    let (offer, _) = change.apply().unwrap();
    let offer_sdp = offer.to_sdp_string().replace(
        "a=rtcp-mux\r\n",
        "a=rtcp-mux\r\na=rtcp-xr:rcvr-rtt=unknown stat-summary=loss\r\n",
    );
    let offer = SdpOffer::from_sdp_string(&offer_sdp).unwrap();
    assert_eq!(offer.to_sdp_string(), offer_sdp);
    let answer = answerer
        .sdp_api()
        .accept_offer(offer)
        .unwrap()
        .to_sdp_string();
    assert_eq!(answer.matches(CAPABILITY).count(), 1);
    assert!(!answer.contains("rcvr-rtt=unknown"));
    assert!(!answer.contains("stat-summary"));
    let application = answer.split("m=application").nth(1).unwrap();
    assert!(!application.contains("a=rtcp-xr:"));
}
