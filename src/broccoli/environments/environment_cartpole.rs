use std::f64::consts::PI;

use crate::broccoli::broccoli_helper_functions::brocolli_within_range;

use super::environment::{Environment, EnvironmentInfo, Interval};

pub struct EnvironmentCartPole {
    //state
    cart_position: f64,
    cart_velocity: f64,
    pole_angle: f64,
    pole_velocity: f64,
    //constants
    gravity: f64,
    //cart_mass: f64,
    pole_mass: f64,
    total_mass: f64,
    polemass_length: f64,
    pole_length: f64,
    force_magnitude: f64,
    time_unit: f64,
}

impl EnvironmentCartPole {
    pub fn new() -> EnvironmentCartPole {
        let gravity = 9.8;
        let cart_mass = 1.0;
        let pole_mass = 0.1;
        let total_mass = pole_mass + cart_mass;
        let pole_length = 0.5;
        let polemass_length = pole_mass * pole_length;
        let force_magnitude = 10.0;
        let time_unit = 0.02;

        EnvironmentCartPole {
            cart_position: 0.0,
            cart_velocity: 0.05,
            pole_angle: 0.05,
            pole_velocity: 0.05,
            gravity,
            //cart_mass,
            pole_mass,
            total_mass,
            polemass_length,
            pole_length,
            force_magnitude,
            time_unit,
        }
    }
}

impl Environment for EnvironmentCartPole {
    //taken from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    fn apply_action(&mut self, action: usize) {
        let force = if action == 1 {
            self.force_magnitude
        } else {
            -self.force_magnitude
        };

        let temp = (force
            + self.polemass_length * self.pole_velocity.powi(2) * self.pole_angle.sin())
            / self.total_mass;
        let pole_acceleration = (self.gravity * self.pole_angle.sin()
            - self.pole_angle.cos() * temp)
            / (self.pole_length
                * (4.0 / 3.0 - self.pole_mass * self.pole_angle.cos().powi(2) / self.total_mass));

        let cart_acceleration = temp
            - self.polemass_length * pole_acceleration * self.pole_angle.cos() / self.total_mass;

        //Euler kinematics integrator
        self.cart_position += self.time_unit * self.cart_velocity;
        self.cart_velocity += self.time_unit * cart_acceleration;
        self.pole_angle += self.time_unit * self.pole_velocity;
        self.pole_velocity += self.time_unit * pole_acceleration;
    }

    fn observe_state(&self) -> Vec<f64> {
        vec![
            self.cart_position,
            self.cart_velocity,
            self.pole_angle,
            self.pole_velocity,
        ]
    }

    fn is_at_terminal_state(&self) -> bool {
        let angle_constant = 12.0 * 2.0 * PI / 360.0;
        !(brocolli_within_range(self.cart_position, -2.4, 2.4)
            && brocolli_within_range(self.pole_angle, -angle_constant, angle_constant))
    }

    fn reset(&mut self, initial_state: &[f64]) {
        self.cart_position = initial_state[0];
        self.cart_velocity = initial_state[1];
        self.pole_angle = initial_state[2];
        self.pole_velocity = initial_state[3];
    }

    fn environment_info(&self) -> EnvironmentInfo {
        let intervals: Vec<Interval> = vec![
            Interval {
                name: "Cart Position".to_string(),
                min: -2.4,
                max: 2.4,
            },
            Interval {
                name: "Cart Velocity".to_string(),
                min: -1.0,
                max: 1.0,
            },
            Interval {
                name: "Pole Angle".to_string(),
                min: -0.418,
                max: 0.418,
            },
            Interval {
                name: "Pole Velocity".to_string(),
                min: -1.0,
                max: 1.0,
            },
        ];
        EnvironmentInfo::new(intervals, 2)
    }
}

#[cfg(test)]
mod tests {
    use crate::broccoli::environments::environment::Environment;

    use super::EnvironmentCartPole;

    #[test]
    fn test() {
        let mut env = EnvironmentCartPole::new();

        env.reset(&vec![0.0, 0.05, 0.05, 0.05]);

        println!("initial state: {:?}", env.observe_state());
        for i in 0..1 {
            let action = 1; // = if env.observe_state()[3] >= -0.1 { 1 } else { 0 };
            println!("action: {}", action);
            env.apply_action(action);
            println!("{}: {:?}", i, env.observe_state());

            let action = 1; // = if env.observe_state()[3] >= -0.1 { 1 } else { 0 };
            println!("action: {}", action);
            env.apply_action(action);
            println!("{}: {:?}", i, env.observe_state());

            if env.is_at_terminal_state() {
                println!("Done at {} iterations", i);
                break;
            }

            let action = 0; // = if env.observe_state()[3] >= -0.1 { 1 } else { 0 };
            println!("action: {}", action);
            env.apply_action(action);
            println!("{}: {:?}", i, env.observe_state());

            if env.is_at_terminal_state() {
                println!("Done at {} iterations", i);
                break;
            }
        }
    }
}
