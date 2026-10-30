//! Virtual association tests exercise the transport config used by each role.

use super::*;
use str0m_proto::DATAGRAM_MTU_TARGET;

struct Datagram {
    at: Instant,
    to: usize,
    bytes: Vec<u8>,
}

struct Pair {
    peers: [RtcSctp; 2],
    now: Instant,
    delay: Duration,
    network: Vec<Datagram>,
    received: [Vec<Vec<u8>>; 2],
    drop_sender: usize,
    drop_tsn: Option<u32>,
    drops_left: usize,
    attempts: Vec<Instant>,
}

/// Parse DATA TSNs without depending on private sctp-proto packet types.
fn data_tsns(packet: &[u8]) -> Vec<u32> {
    let mut offset = 12;
    let mut result = Vec::new();
    while offset + 4 <= packet.len() {
        let length = u16::from_be_bytes([packet[offset + 2], packet[offset + 3]]) as usize;
        assert!(length >= 4 && offset + length <= packet.len());
        if packet[offset] == 0 {
            assert!(length >= 16);
            result.push(u32::from_be_bytes(
                packet[offset + 4..offset + 8].try_into().unwrap(),
            ));
        }
        offset += (length + 3) & !3;
    }
    result
}

impl Pair {
    fn new(snap: bool, delay_ms: u64) -> Self {
        let now = Instant::now();
        let mut peers = [
            RtcSctp::new(DATAGRAM_MTU_TARGET),
            RtcSctp::new(DATAGRAM_MTU_TARGET),
        ];
        if snap {
            let mut client = SctpInitData::new();
            let mut server = SctpInitData::new();
            let client_init = client.local_init_chunk().unwrap();
            let server_init = server.local_init_chunk().unwrap();
            client.set_remote_init_chunk(server_init);
            server.set_remote_init_chunk(client_init);
            peers[0].init(true, now, Some(client), None).unwrap();
            peers[1].init(false, now, Some(server), None).unwrap();
        } else {
            peers[0].init(true, now, None, None).unwrap();
            peers[1].init(false, now, None, None).unwrap();
        }
        let mut pair = Self {
            peers,
            now,
            delay: Duration::from_millis(delay_ms),
            network: Vec::new(),
            received: [Vec::new(), Vec::new()],
            drop_sender: 0,
            drop_tsn: None,
            drops_left: 0,
            attempts: Vec::new(),
        };
        pair.advance(2000);
        for peer in &mut pair.peers {
            assert_eq!(peer.state, RtcSctpState::Established);
            peer.open_stream(
                0,
                ChannelConfig {
                    label: "reliable-control".into(),
                    ordered: true,
                    reliability: Reliability::Reliable,
                    negotiated: Some(0),
                    protocol: String::new(),
                },
            );
        }
        pair.advance(1);
        pair
    }

    fn pump(&mut self) {
        for from in 0..2 {
            while let Some(event) = self.peers[from].do_poll() {
                match event {
                    SctpEvent::Transmit { packets } => {
                        for bytes in packets {
                            let tsns = data_tsns(&bytes);
                            if from == self.drop_sender
                                && self.drops_left > 0
                                && self.drop_tsn.is_none()
                            {
                                self.drop_tsn = tsns.first().copied();
                            }
                            if from == self.drop_sender
                                && self.drop_tsn.is_some_and(|tsn| tsns.contains(&tsn))
                            {
                                self.attempts.push(self.now);
                                if self.drops_left > 0 {
                                    self.drops_left -= 1;
                                    continue;
                                }
                            }
                            self.network.push(Datagram {
                                at: self.now + self.delay,
                                to: 1 - from,
                                bytes,
                            });
                        }
                    }
                    SctpEvent::Data { data, .. } => self.received[from].push(data),
                    _ => {}
                }
            }
        }
    }

    fn advance(&mut self, millis: u64) {
        for _ in 0..millis {
            self.now += Duration::from_millis(1);
            let mut index = 0;
            while index < self.network.len() {
                if self.network[index].at <= self.now {
                    let packet = self.network.remove(index);
                    self.peers[packet.to].handle_input(self.now, &packet.bytes);
                } else {
                    index += 1;
                }
            }
            for peer in &mut self.peers {
                peer.handle_timeout(self.now);
            }
            self.pump();
        }
    }

    fn send(&mut self, from: usize, bytes: &[u8]) {
        assert_eq!(
            self.peers[from].write(0, false, bytes).unwrap(),
            bytes.len()
        );
        self.pump();
    }

    fn timeouts(&self, from: usize) -> u64 {
        self.peers[from]
            .assoc
            .as_ref()
            .unwrap()
            .stats()
            .get_num_t3timeouts()
    }

    fn warm_up(&mut self, from: usize) {
        for _ in 0..30 {
            self.send(from, b"warmup");
            self.advance(600);
        }
        self.received[1 - from].clear();
        assert_eq!(
            self.timeouts(from),
            0,
            "lossless warmup must not retransmit"
        );
    }
}

#[test]
fn webrtc_rto_initial_loss_recovery_in_all_association_paths() {
    for snap in [false, true] {
        for from in 0..2 {
            let mut pair = Pair::new(snap, 5);
            pair.drop_sender = from;
            pair.drops_left = 2;
            pair.send(from, b"first");
            pair.advance(1600);
            assert_eq!(
                pair.received[1 - from],
                [b"first".to_vec()],
                "snap={snap} sender={from}"
            );
            assert_eq!(pair.attempts.len(), 3);
            assert_eq!(
                pair.attempts[1] - pair.attempts[0],
                Duration::from_millis(500)
            );
            assert_eq!(
                pair.attempts[2] - pair.attempts[1],
                Duration::from_millis(1000)
            );
            assert_eq!(pair.timeouts(from), 2);
            pair.advance(4000);
            assert_eq!(
                pair.received[1 - from].len(),
                1,
                "reliable delivery must not duplicate"
            );
        }
    }
}

#[test]
fn webrtc_rto_sampled_loss_recovers_ordered_messages_once() {
    for snap in [false, true] {
        for from in 0..2 {
            let mut pair = Pair::new(snap, 5);
            pair.warm_up(from);
            pair.drop_sender = from;
            pair.drops_left = 2;
            pair.send(from, b"first");
            pair.advance(1);
            pair.send(from, b"second");
            pair.advance(1300);
            assert_eq!(
                pair.received[1 - from],
                [b"first".to_vec(), b"second".to_vec()],
                "snap={snap} sender={from}"
            );
            assert_eq!(pair.attempts.len(), 3);
            assert_eq!(
                pair.attempts[1] - pair.attempts[0],
                Duration::from_millis(400)
            );
            assert_eq!(
                pair.attempts[2] - pair.attempts[1],
                Duration::from_millis(800)
            );
            assert_eq!(pair.timeouts(from), 2);
            pair.advance(4000);
            assert_eq!(pair.received[1 - from].len(), 2);
        }
    }
}

#[test]
fn webrtc_rto_clean_delayed_path_does_not_retransmit() {
    for snap in [false, true] {
        for from in 0..2 {
            let mut pair = Pair::new(snap, 150);
            pair.warm_up(from);
            pair.send(from, b"isolated-control");
            pair.advance(1000);
            assert_eq!(pair.received[1 - from], [b"isolated-control".to_vec()]);
            assert_eq!(
                pair.timeouts(from),
                0,
                "no loss at300ms RTT and delayed ACK"
            );
        }
    }
}

#[test]
fn webrtc_rto_path_delay_increase_recovers() {
    for snap in [false, true] {
        for from in 0..2 {
            let mut pair = Pair::new(snap, 5);
            pair.warm_up(from);
            pair.delay = Duration::from_millis(150);
            pair.send(from, b"isolated-control");
            pair.advance(1000);
            assert_eq!(pair.received[1 - from], [b"isolated-control".to_vec()]);
            // One timeout may follow a sudden delay increase. Later messages
            // must obtain a clean RTT sample instead of repeating that timeout.
            for index in 0..30 {
                pair.send(from, format!("after-transition-{index}").as_bytes());
                pair.advance(1000);
            }
            let expected = std::iter::once(b"isolated-control".to_vec())
                .chain((0..30).map(|index| format!("after-transition-{index}").into_bytes()))
                .collect::<Vec<_>>();
            assert_eq!(pair.received[1 - from], expected);
            assert!(
                pair.timeouts(from) <= 1,
                "snap={snap} sender={from}: repeated lossless T3 timeouts: {}",
                pair.timeouts(from)
            );
        }
    }
}
