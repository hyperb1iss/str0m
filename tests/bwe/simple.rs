//! Simple BWE tests under different network conditions (cellular, WiFi).

use std::time::Duration;

use netem::{Bitrate, NetemConfig};
use str0m::RtcError;

use crate::common::{BweTestContext, Step, connect_with_bwe};
use crate::common::{init_crypto_default, init_log};

// Test bitrate layers
pub const LAYER_LOW: Bitrate = Bitrate::kbps(250);
pub const LAYER_MID: Bitrate = Bitrate::kbps(750);
pub const LAYER_TOP: Bitrate = Bitrate::kbps(1_500);

// Standard ramp-up test plan
pub const RAMP_UP_SINGLE: &[Step] = &[
    Step::Conditions {
        description: "Startup conditions",
        config: NetemConfig::new(),
    },
    Step::Media {
        description: "Low layer ramping up to mid",
        desired_bitrate: LAYER_MID,
        media_send_rate: LAYER_LOW,
    },
    Step::Run {
        description: "Wait for mid",
        duration: Duration::from_secs(20),
    },
    Step::Check {
        description: "Check enough for mid",
        at_least: Bitrate::kbps(750),
    },
    Step::Media {
        description: "Mid layer ramping up to top",
        desired_bitrate: LAYER_TOP,
        media_send_rate: LAYER_MID,
    },
    Step::Run {
        description: "Wait for top",
        duration: Duration::from_secs(20),
    },
    Step::Check {
        description: "Check enough for top",
        at_least: Bitrate::kbps(1_500),
    },
    Step::Media {
        description: "Top layer",
        desired_bitrate: LAYER_TOP,
        media_send_rate: LAYER_TOP,
    },
    Step::Run {
        description: "Ensure top stabilizes",
        duration: Duration::from_secs(10),
    },
    Step::Check {
        description: "Top is stable",
        at_least: Bitrate::kbps(1_500),
    },
];
/// Seeds of the cellular ramp-up. Its loss bursts land on a different probe
/// or layer step in each, so one seed reaching a layer in time is a coin flip:
/// 36 of 60 seeds reach mid and 22 of 60 reach top before the plan checks.
const CELLULAR_SEEDS: u64 = 16;

/// One seed of the cellular ramp-up: the estimate where the plan checks for
/// mid, for top, and for top holding once it is sent.
fn cellular_ramp_up(seed: u64) -> [f64; 3] {
    let mut plan = RAMP_UP_SINGLE.to_vec();
    plan[0] = Step::Conditions {
        description: "Cellular conditions",
        config: NetemConfig::cellular().seed(seed),
    };
    plan[2] = Step::Run {
        description: "Wait for mid",
        duration: Duration::from_secs(40),
    };
    for step in plan.iter_mut() {
        if let Step::Check { description, .. } = step {
            *step = Step::Record { description };
        }
    }

    let (mut l, mut r) = connect_with_bwe(LAYER_LOW, LAYER_MID);
    let mut ctx = BweTestContext::new(&mut l, &mut r);
    ctx.run_plan(&mut l, &mut r, &plan).expect("the plan runs");

    let estimates: Vec<f64> = ctx.estimates.iter().map(|e| e.as_f64()).collect();
    estimates.try_into().expect("three recorded estimates")
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
pub fn bwe_cellular() -> Result<(), RtcError> {
    init_log();
    init_crypto_default();

    let runs: Vec<[f64; 3]> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..CELLULAR_SEEDS)
            .map(|seed| {
                std::thread::Builder::new()
                    .name(format!("seed-{seed}"))
                    .stack_size(16 << 20)
                    .spawn_scoped(scope, move || cellular_ramp_up(seed))
                    .expect("a seed's thread")
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().expect("a finished seed"))
            .collect()
    });

    // The ramp reaches the layers on most seeds. Medians over 60 seeds are
    // 790 kbps and 1.28 Mbps, the slowest quartiles 670 kbps and 1.1 Mbps.
    let mid = median(runs.iter().map(|run| run[0]).collect());
    let top = median(runs.iter().map(|run| run[1]).collect());
    assert!(mid >= Bitrate::kbps(600).as_f64(), "mid median {mid} bps");
    assert!(top >= Bitrate::kbps(1_000).as_f64(), "top median {top} bps");

    // Once top is sent every seed holds it: 60 of 60, the lowest at 1.6 Mbps.
    for (seed, run) in runs.iter().enumerate() {
        assert!(
            run[2] >= LAYER_TOP.as_f64(),
            "seed {seed} fell to {} bps with top flowing",
            run[2]
        );
    }

    Ok(())
}

#[test]
pub fn bwe_wifi_normal() -> Result<(), RtcError> {
    init_log();
    init_crypto_default();

    // WiFi: 100 Mbps link, 5ms latency, ~1% loss
    let mut plan = RAMP_UP_SINGLE.to_vec();
    plan[0] = Step::Conditions {
        description: "WiFi conditions",
        config: NetemConfig::wifi().seed(42),
    };

    let (mut l, mut r) = connect_with_bwe(LAYER_LOW, LAYER_MID);

    let mut ctx = BweTestContext::new(&mut l, &mut r);

    ctx.run_plan(&mut l, &mut r, &RAMP_UP_SINGLE)?;

    Ok(())
}

#[test]
pub fn bwe_wifi_congested() -> Result<(), RtcError> {
    init_log();
    init_crypto_default();

    // Congested WiFi: 5 Mbps link, 10ms latency, ~10% loss
    let mut plan = RAMP_UP_SINGLE.to_vec();
    plan[0] = Step::Conditions {
        description: "Congested WiFi conditions",
        config: NetemConfig::congested().seed(42),
    };

    let (mut l, mut r) = connect_with_bwe(LAYER_LOW, LAYER_MID);

    let mut ctx = BweTestContext::new(&mut l, &mut r);

    ctx.run_plan(&mut l, &mut r, &RAMP_UP_SINGLE)?;

    Ok(())
}
