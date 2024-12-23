use std::{fs::File, io::Write};

use clap::Parser;

use crate::broccoli::{
    broccoli_helper_functions::extract_initial_states,
    environments::environment::Interval,
    runners::{
        cart_pole_runner::run_cart_pole, mountain_car_runner::run_mountain_car,
        pendulum_runner::run_pendulum,
    },
};

use rand::{rngs::SmallRng, Rng, SeedableRng};

mod broccoli;

#[derive(Clone, Debug)]
enum EnvironmentType {
    MountainCar,
    CartPole,
    Pendulum,
}

//define the parameters for the problem
//  these can be given via the command line, otherwise the default value will be used
//  example: cargo run --release -- --env mc -d 2 -i 100 -x -0.4 0.1 -p 0.1 0.1 -> mountain car, depth 2, 100 iterations for simulation, initial state [-0.4, 0.1] with discretisation 0.1 for both state components
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
#[clap(allow_negative_numbers = true)]
struct Args {
    #[arg(long = "env", value_parser = parser_environment_type)]
    environment_type: CliArg<EnvironmentType>,

    /// The maximum depth of the tree
    #[arg(short = 'd', long = "depth")]
    depth: u32,

    /// The maximum number of nodes
    #[arg(short = 'n', long = "num-nodes")]
    num_nodes: u32,

    /// The number of iterations used for the simulation
    #[arg(short = 'i', long = "num-iters")]
    num_simulation_iterations: u32,

    /// Technique to skip redundant predicates that would not influence the simulation outcome
    #[arg(long = "predicate-reasoning", value_parser = parser_bool_parameter, default_value_t = true.into())]
    use_predicate_reasoning: CliArg<bool>,

    #[arg(short = 'x', long = "initial-state-values", num_args = 1..,)]
    initial_states_flattened: Vec<f64>,

    /// The increment values between two predicates, i.e., predicates are of the form [x_i >= min-value_i + inc * k], where k is an integer
    #[arg(short = 'p', long = "predicate-increment", num_args = 1..)]
    predicate_increments: Vec<f64>,

    /// Used for NeurIPS experiments. Given as a pair of values "[seed] [num_initial_states]", where the states are randomly sampled depending on the benchmark
    #[arg(long = "NeurIPS", num_args = 2)]
    neurips_parameters: Vec<u64>,
}

#[derive(Debug, Clone)]
struct CliArg<T> {
    inner: T,
}

fn parser_environment_type(s: &str) -> Result<CliArg<EnvironmentType>, String> {
    let s_l = s.to_lowercase();
    if s_l == "mc" || s_l == "mountain_car" || s_l == "mountain-car" {
        Ok(EnvironmentType::MountainCar.into())
    } else if s_l == "cp" || s_l == "cartpole" || s_l == "cart-pole" {
        Ok(EnvironmentType::CartPole.into())
    } else if s_l == "pen" || s_l == "pendulum" {
        Ok(EnvironmentType::Pendulum.into())
    } else {
        Err(format!("'{s}' is not recognised as an environment."))
    }
}

fn parser_bool_parameter(s: &str) -> Result<CliArg<bool>, String> {
    if s == "1" || s.to_lowercase() == "true" {
        Ok(true.into())
    } else if s == "0" || s.to_lowercase() == "false" {
        Ok(false.into())
    } else {
        Err(format!("'{s}' is not valid input for a Boolean parameter."))
    }
}

impl<T> From<T> for CliArg<T> {
    fn from(value: T) -> Self {
        CliArg { inner: value }
    }
}

impl std::fmt::Display for CliArg<bool> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

//todo:
//  + tests
//  + incrementality of the simulation:
//      + do not run from scratch, but rather revert to some point in time when the new changes make a difference
//      + ...maybe this does not make a big difference, but it might

//clean up tests...

//backjump seems useful -> if the predicate node you are changing has never been used, go to the previous node
//opening new node, could use thresholds of parent nodes to determine the first starting point? But only if it has a direct parent with same feature?
//symmetries, where using more predicates is equivalent to using a smaller number of predicates. (x >= 2 root, x >= 5 left child)
//  + also maybe cases where this is not logically implied, but based on simulation runs happens to be the case
//add checks to ensure that increment is always increasing -> strange behaviour detected where thresholds bounce from 0.5, 0.7, and then 0.6

fn get_environment_state_variable_ranges(environment_type: &EnvironmentType) -> Vec<Interval> {
    match environment_type {
        EnvironmentType::MountainCar => {
            vec![
                Interval {
                    name: "Position".to_string(),
                    min: -0.6,
                    max: -0.4,
                },
                Interval {
                    name: "Velocity".to_string(),
                    min: 0.0,
                    max: 0.0,
                },
            ]
        }
        EnvironmentType::CartPole => {
            vec![
                Interval {
                    name: "Cart Position".to_string(),
                    min: -0.05,
                    max: 0.05,
                },
                Interval {
                    name: "Cart Velocity".to_string(),
                    min: -0.05,
                    max: 0.05,
                },
                Interval {
                    name: "Pole Angle".to_string(),
                    min: -0.05,
                    max: 0.05,
                },
                Interval {
                    name: "Pole Velocity".to_string(),
                    min: -0.05,
                    max: 0.05,
                },
            ]
        }
        EnvironmentType::Pendulum => {
            vec![
                Interval {
                    name: "Angle".to_string(),
                    min: -0.8,
                    max: -0.5,
                },
                Interval {
                    name: "Angular Velocity".to_string(),
                    min: -0.2,
                    max: 0.2,
                },
            ]
        }
    }
}

fn neurips_script_experiment1() {
    let mut script: String = String::new();

    let mut file = File::create("experiments1.sh").expect("Problem writing to file?");

    let num_runs = 10;

    //SINGLE INITIAL STATE
    for seed in 0..num_runs {
        let name = "cp";

        script += &format!("echo \"exp1 {name} {seed} yes\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.1 0.1 0.1 0.1 --NeurIPS {seed} 1 --predicate-reasoning 1 > exp1_single_{name}_{seed}_yes.txt\n");

        script += &format!("echo \"exp1 {name} {seed} no\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.1 0.1 0.1 0.1 --NeurIPS {seed} 1 --predicate-reasoning 0 > exp1_single_{name}_{seed}_no.txt\n");
    }

    for seed in 0..num_runs {
        let name = "mc";

        script += &format!("echo \"exp1 {name} {seed} yes\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.05 0.005 --NeurIPS {seed} 1 --predicate-reasoning 1 > exp1_single_{name}_{seed}_yes.txt\n");

        script += &format!("echo \"exp1 {name} {seed} no\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.05 0.005 --NeurIPS {seed} 1 --predicate-reasoning 0 > exp1_single_{name}_{seed}_no.txt\n");
    }

    for seed in 0..num_runs {
        let name = "pen";

        script += &format!("echo \"exp1 {name} {seed} yes\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.2 0.2 --NeurIPS {seed} 1 --predicate-reasoning 1 > exp1_single_{name}_{seed}_yes.txt\n");

        script += &format!("echo \"exp1 {name} {seed} no\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.2 0.2 --NeurIPS {seed} 1 --predicate-reasoning 0 > exp1_single_{name}_{seed}_no.txt\n");
    }

    //MULTIPLE INITIAL STATES
    for seed in 0..num_runs {
        let name = "cp";

        script += &format!("echo \"exp1 {name} {seed} yes\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.1 0.1 0.1 0.1 --NeurIPS {seed} 100 --predicate-reasoning 1 > exp1_multiple_{name}_{seed}_yes.txt\n");

        script += &format!("echo \"exp1 {name} {seed} no\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.1 0.1 0.1 0.1 --NeurIPS {seed} 100 --predicate-reasoning 0 > exp1_multiple_{name}_{seed}_no.txt\n");
    }

    for seed in 0..num_runs {
        let name = "mc";

        script += &format!("echo \"exp1 {name} {seed} yes\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.05 0.005 --NeurIPS {seed} 100 --predicate-reasoning 1 > exp1_multiple_{name}_{seed}_yes.txt\n");

        script += &format!("echo \"exp1 {name} {seed} no\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.05 0.005 --NeurIPS {seed} 100 --predicate-reasoning 0 > exp1_multiple_{name}_{seed}_no.txt\n");
    }

    for seed in 0..num_runs {
        let name = "pen";

        script += &format!("echo \"exp1 {name} {seed} yes\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.2 0.2 --NeurIPS {seed} 100 --predicate-reasoning 1 > exp1_multiple_{name}_{seed}_yes.txt\n");

        script += &format!("echo \"exp1 {name} {seed} no\" \n");
        script += &format!("./broccoli.exe --env {name} --depth 2 --num-iters 10000 --predicate-increment 0.2 0.2 --NeurIPS {seed} 100 --predicate-reasoning 0 > exp1_multiple_{name}_{seed}_no.txt\n");
    }

    file.write(script.as_bytes())
        .expect("Writing as bytes failed?");

    panic!();
}

fn neurips_script_experiment_scale_predicates() {
    let mut script: String = String::new();

    let mut file =
        File::create("experiments_scale_predicates.sh").expect("Problem writing to file?");

    let num_runs = 10;

    //SINGLE INITIAL STATE

    for depth in 2..=3 {
        for seed in 0..num_runs {
            let name = "mc";

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 5\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.36 0.028 --NeurIPS {seed} 1 > expPred_5_single_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 10\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.18 0.014 --NeurIPS {seed} 1 > expPred_10_single_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 15\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.12 0.01 --NeurIPS {seed} 1 > expPred_15_single_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 20\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.09 0.007 --NeurIPS {seed} 1 > expPred_20_single_{name}_{seed}_d={depth}.txt\n");

            let name = "cp";

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 5\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.96 0.4 0.16 0.4 --NeurIPS {seed} 1 > expPred_5_single_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 10\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.48 0.2 0.33 0.2 --NeurIPS {seed} 1 > expPred_10_single_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 15\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.32 0.13 0.08 0.13 --NeurIPS {seed} 1 > expPred_15_single_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 20\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.24 0.1 0.04 0.1 --NeurIPS {seed} 1 > expPred_20_single_{name}_{seed}_d={depth}.txt\n");

            let name = "pen";

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 5\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.4 3.2 --NeurIPS {seed} 1 > expPred_5_single_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 10\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.2 1.6 --NeurIPS {seed} 1 > expPred_10_single_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 15\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.13 1.0 --NeurIPS {seed} 1 > expPred_15_single_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred depth {depth} {name} {seed} 20\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.1 0.8 --NeurIPS {seed} 1 > expPred_20_single_{name}_{seed}_d={depth}.txt\n");
        }

        //MULTIPLE INITIAL STATE

        for seed in 0..num_runs {
            let name = "mc";

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 5\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.36 0.028 --NeurIPS {seed} 100 > expPred_5_multiple_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 10\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.18 0.014 --NeurIPS {seed} 100 > expPred_10_multiple_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 15\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.12 0.01 --NeurIPS {seed} 100 > expPred_15_multiple_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 20\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.09 0.007 --NeurIPS {seed} 100 > expPred_20_multiple_{name}_{seed}_d={depth}.txt\n");

            let name = "cp";

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 5\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.96 0.4 0.16 0.4 --NeurIPS {seed} 100 > expPred_5_multiple_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 10\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.48 0.2 0.33 0.2 --NeurIPS {seed} 100 > expPred_10_multiple_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 15\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.32 0.13 0.08 0.13 --NeurIPS {seed} 100 > expPred_15_multiple_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 20\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.24 0.1 0.04 0.1 --NeurIPS {seed} 100 > expPred_20_multiple_{name}_{seed}_d={depth}.txt\n");

            let name = "pen";

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 5\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.4 3.2 --NeurIPS {seed} 100 > expPred_5_multiple_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 10\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.2 1.6 --NeurIPS {seed} 100 > expPred_10_multiple_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 15\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.13 1.0 --NeurIPS {seed} 100 > expPred_15_multiple_{name}_{seed}_d={depth}.txt\n");

            script += &format!("echo \"exp_pred multi depth {depth} {name} {seed} 20\" \n");
            script += &format!("./broccoli.exe --env {name} --depth {depth} --num-iters 10000 --predicate-increment 0.1 0.8 --NeurIPS {seed} 100 > expPred_20_multiple_{name}_{seed}_d={depth}.txt\n");
        }
    }

    file.write(script.as_bytes())
        .expect("Writing as bytes failed?");

    panic!();
}

fn neurips_script_experiment_num_nodes() {
    let mut script: String = String::new();

    let mut file =
        File::create("experiments_scale_num_nodes.sh").expect("Problem writing to file?");

    let num_runs = 10;

    /*for seed in 0..num_runs {
        for num_nodes in 3..=7 {
            let name = "mc";

            script += &format!("echo \"mc {num_nodes} {name} {seed} 10\" \n");
            script += &format!("./broccoli2.exe --env {name} --depth 3 --num-nodes {num_nodes} --num-iters 10000 --predicate-increment 0.18 0.014 --NeurIPS {seed} 1 > expSize_10_single_{name}_{seed}_n={num_nodes}.txt\n");

            let name = "cp";

            script += &format!("echo \"cp {num_nodes} {name} {seed} 5\" \n");
            script += &format!("./broccoli2.exe --env {name} --depth 3 --num-nodes {num_nodes} --num-iters 10000 --predicate-increment 0.96 0.4 0.16 0.4 --NeurIPS {seed} 1 > expSize_5_single_{name}_{seed}_n={num_nodes}.txt\n");

            let name = "pen";

            script += &format!("echo \"pen {num_nodes} {name} {seed} 10\" \n");
            script += &format!("./broccoli2.exe --env {name} --depth 3 --num-nodes {num_nodes} --num-iters 10000 --predicate-increment 0.2 1.6 --NeurIPS {seed} 1 > expSize_10_single_{name}_{seed}_n={num_nodes}.txt\n");
        }
    }*/

    for seed in 0..num_runs {
        for num_nodes in 3..=7 {
            let name = "mc";

            script += &format!("echo \"mc {num_nodes} {name} {seed} 10\" \n");
            script += &format!("./broccoli2.exe --env {name} --depth 3 --num-nodes {num_nodes} --num-iters 10000 --predicate-increment 0.18 0.014 --NeurIPS {seed} 100 > expSize_10_multiple_{name}_{seed}_n={num_nodes}.txt\n");

            let name = "cp";

            script += &format!("echo \"cp {num_nodes} {name} {seed} 5\" \n");
            script += &format!("./broccoli2.exe --env {name} --depth 3 --num-nodes {num_nodes} --num-iters 10000 --predicate-increment 0.96 0.4 0.16 0.4 --NeurIPS {seed} 100 > expSize_5_multiple_{name}_{seed}_n={num_nodes}.txt\n");

            let name = "pen";

            script += &format!("echo \"pen {num_nodes} {name} {seed} 10\" \n");
            script += &format!("./broccoli2.exe --env {name} --depth 3 --num-nodes {num_nodes} --num-iters 10000 --predicate-increment 0.2 1.6 --NeurIPS {seed} 100 > expSize_10_multiple_{name}_{seed}_n={num_nodes}.txt\n");
        }
    }

    file.write(script.as_bytes())
        .expect("Writing as bytes failed?");

    panic!();
}

fn main() {
    //neurips_script_experiment1();
    //neurips_script_experiment_scale_predicates();
    //neurips_script_experiment_num_nodes();

    //2_u32.pow(decision_tree_depth) - 1

    let args = Args::parse();

    println!("Depth: {}", args.depth);

    let num_state_variables = match args.environment_type.inner {
        EnvironmentType::MountainCar => 2,
        EnvironmentType::CartPole => 4,
        EnvironmentType::Pendulum => 2,
    };

    let initial_states = if args.neurips_parameters.is_empty() {
        extract_initial_states(&args.initial_states_flattened, num_state_variables)
    } else {
        assert!(
            args.neurips_parameters.len() == 2,
            "Expected two values for the NeurIPS experiments."
        );
        //depending on the environment, create a vector of state ranges
        //  the initial states will be randomly sampled within these ranges
        let state_variable_ranges =
            get_environment_state_variable_ranges(&args.environment_type.inner);
        let seed = args.neurips_parameters[0];
        println!("Seed: {seed}");

        let num_initial_states = args.neurips_parameters[1];
        println!("Num initial states: {num_initial_states}");

        let mut random_generator = SmallRng::seed_from_u64(seed);

        let mut initial_states: Vec<Vec<f64>> = vec![];
        for _i in 0..num_initial_states {
            let mut state: Vec<f64> = vec![];
            for variable_range in &state_variable_ranges {
                let mut val = random_generator.gen_range(variable_range.min..=variable_range.max);

                //rounding to two decimal places for simplicity
                val *= 1000.0;
                val = ((val as i64) as f64) / 1000.0;

                state.push(val);
            }
            initial_states.push(state);
        }
        initial_states
    };

    if initial_states.len() <= 100 {
        for state in initial_states.iter().enumerate() {
            println!("State {}: {:?}", state.0, state.1);
        }
    }

    match args.environment_type.inner {
        EnvironmentType::MountainCar => {
            run_mountain_car(
                args.depth,
                args.num_nodes,
                args.num_simulation_iterations,
                &initial_states,
                &args.predicate_increments,
                args.use_predicate_reasoning.inner,
            );
        }
        EnvironmentType::CartPole => {
            run_cart_pole(
                args.depth,
                args.num_nodes,
                args.num_simulation_iterations,
                &initial_states,
                &args.predicate_increments,
                args.use_predicate_reasoning.inner,
            );
        }
        EnvironmentType::Pendulum => run_pendulum(
            args.depth,
            args.num_nodes,
            args.num_simulation_iterations,
            &initial_states,
            &args.predicate_increments,
            args.use_predicate_reasoning.inner,
        ),
    }
}
