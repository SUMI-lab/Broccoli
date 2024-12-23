use super::environment::{Environment, EnvironmentInfo, Interval};

pub struct EnvironmentMountainCar {
    state: Vec<f64>, //[0] is the position, [1] is the velocity
    force: f64,
    gravity: f64,
}

impl EnvironmentMountainCar {
    pub fn new() -> EnvironmentMountainCar {
        EnvironmentMountainCar {
            state: vec![0.0, 0.0],
            force: 0.001,
            gravity: 0.0025,
        }
    }

    fn compute_velocity_change(&self, action: usize) -> f64 {
        let action_contribution: f64 = match action {
            0 => -self.force,
            1 => self.force,
            _ => unreachable!(),
        };
        let gravity_contribution = (3.0 * self.state[0]).cos() * self.gravity;
        action_contribution - gravity_contribution
    }
}

impl Environment for EnvironmentMountainCar {
    fn apply_action(&mut self, action: usize) {
        let velocity_change: f64 = self.compute_velocity_change(action);

        //update velocity
        self.state[1] += velocity_change;
        self.state[1] = self.state[1].clamp(-0.7, 0.7);

        //update position
        self.state[0] += self.state[1];
        self.state[0] = self.state[0].clamp(-1.2, 0.6);

        //if the state is in the extreme left, clamp the velocity to zero
        //  we do this to faithfully replicate the gymnasium environment
        if self.state[0] <= -1.2 && self.state[1] < 0.0 {
            self.state[1] = 0.0;
        }
    }

    fn observe_state(&self) -> Vec<f64> {
        self.state.clone()
    }

    fn is_at_terminal_state(&self) -> bool {
        self.state[0] >= 0.5
    }

    fn reset(&mut self, initial_state: &[f64]) {
        self.state = initial_state.to_vec();
    }

    fn environment_info(&self) -> EnvironmentInfo {
        let intervals: Vec<Interval> = vec![
            Interval {
                name: "Position".to_string(),
                min: -1.2,
                max: 0.6,
            },
            Interval {
                name: "Velocity".to_string(),
                min: -0.07,
                max: 0.07,
            },
        ];
        EnvironmentInfo::new(intervals, 2)
    }
}

#[cfg(test)]
mod tests {
    /*use crate::broccoli::environments::environment::Environment;

    use super::EnvironmentMountainCar;

    #[test]
    fn in_valley() {
        let mut env = EnvironmentMountainCar::new();
        println!("initial state: {:?}", env.observe_state());
        for _ in 0..10 {
            env.apply_action(0);
            println!("s: {:?}", env.observe_state());
        }
    }

    #[test]
    fn in_valley2() {
        let mut env = EnvironmentMountainCar::new();
        println!("initial state: {:?}", env.observe_state());
        for i in 0..100 {
            let action = if env.observe_state()[1] >= 0.0 { 1 } else { 0 };
            env.apply_action(action);
            println!("s: {:?}", env.observe_state());

            if env.is_at_terminal_state() {
                println!("Done at {} iterations", i);
                break;
            }
        }
    }*/
}
