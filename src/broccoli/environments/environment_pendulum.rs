use crate::broccoli::broccoli_helper_functions::brocolli_within_range;

use super::environment::{Environment, EnvironmentInfo, Interval};

//taken from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

pub struct EnvironmentPendulum {
    angle: f64, //in radians
    angular_velocity: f64,

    gravity: f64,
    max_velocity: f64,
    m: f64,
    l: f64,
    time_unit: f64,
}

impl EnvironmentPendulum {
    pub fn new() -> EnvironmentPendulum {
        EnvironmentPendulum {
            angle: 0.0,
            angular_velocity: 0.0,
            gravity: 10.0,
            max_velocity: 8.0,
            m: 1.0,
            l: 1.0,
            time_unit: 0.05,
        }
    }
    /*fn normalise_angle(a: f64) -> f64 {
        let pi = std::f64::consts::PI;
        ((a + pi) % (2.0 * pi)) - pi
    }*/
}

impl Environment for EnvironmentPendulum {
    fn apply_action(&mut self, action: usize) {
        let u: f64 = if action == 0 {
            -2.0
        } else if action == 1 {
            2.0
        } else {
            panic!();
        };
        /*let costs = f64::powi(EnvironmentPendulum::normalise_angle(self.angle), 2)
        + 0.1 * f64::powi(self.angluar_velocity, 2)
        + 0.001 * f64::powi(u, 2);*/
        let new_angluar_velocity = self.angular_velocity
            + (3.0 * self.gravity / (2.0 * self.l) * self.angle.sin()
                + 3.0 / (self.m * f64::powi(self.l, 2)) * u)
                * self.time_unit;

        /*println!(
            "first part: {}",
            3.0 * self.gravity / (2.0 * self.l) * self.angle.sin()
        );

        println!(
            "second part: {}",
            (3.0 / (self.m * f64::powi(self.l, 2)) * u) * self.time_unit
        );*/

        self.angular_velocity = new_angluar_velocity.clamp(-self.max_velocity, self.max_velocity);

        //println!("dot: {}", self.angluar_velocity);

        self.angle += new_angluar_velocity * self.time_unit;
        /*println!("after");
        println!(
            "{} {} {}",
            self.angle.cos(),
            self.angle.sin(),
            self.angluar_velocity
        );
        println!("{}", self.angle);*/
    }

    //possible issue: openAI gym uses f32, and not f64, so there may be a discrepancy
    fn observe_state(&self) -> Vec<f64> {
        vec![self.angle, self.angular_velocity]
    }

    fn is_at_terminal_state(&self) -> bool {
        //the angle is (almost) upright
        //broccoli_greater_or_equal(-0.8, self.angle) || broccoli_greater_or_equal(self.angle, 0.8)
        brocolli_within_range(self.angle, -0.1, 0.1)
            && brocolli_within_range(self.angular_velocity, -0.1, 0.1)
    }

    fn reset(&mut self, initial_state: &[f64]) {
        self.angle = initial_state[0];
        self.angular_velocity = initial_state[1];
    }

    fn environment_info(&self) -> EnvironmentInfo {
        let intervals: Vec<Interval> = vec![
            Interval {
                name: "Angle".to_string(),
                min: -1.0,
                max: 1.0,
            },
            Interval {
                name: "Angular Velocity".to_string(),
                min: -self.max_velocity,
                max: self.max_velocity,
            },
        ];
        EnvironmentInfo::new(intervals, 2)
    }
}

#[cfg(test)]
mod tests {
    use crate::broccoli::environments::environment::Environment;

    use super::EnvironmentPendulum;

    #[test]
    fn test() {
        let mut env = EnvironmentPendulum::new();
        env.reset(&[0.0, 0.0]);

        println!(
            "{} {} {}",
            env.angle.cos(),
            env.angle.sin(),
            env.angular_velocity
        );
        println!("{}", env.angle);
        for _i in 0..11 {
            env.apply_action(0);
        }
    }

    #[test]
    fn specific_state() {
        let mut env = EnvironmentPendulum::new();
        env.reset(&[2.2921520451893973, -0.04754959434214606]);
        println!("{}", env.angle);
        env.apply_action(0);
        println!("{}", env.angle);
    }
}
