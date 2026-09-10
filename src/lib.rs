use crate::broccoli::environments::environment::{Environment, EnvironmentInfo};
use numpy::PyArray1;
use pyo3::prelude::*;

mod broccoli;

#[pymodule]
mod broccoli_python {
    use std::time::Duration;

use crate::broccoli::broccoli;
    use crate::broccoli::trees::decision_tree::Node;
    use crate::PythonBasedEnvironment;
    use pyo3::prelude::*;

    #[pyfunction]
    fn run_solver(
        mut env: PythonBasedEnvironment,
        depth: u32,
        num_nodes: u32,
        num_simulation_iterations: u32,
        predicate_increments: Vec<f64>,
        use_predicate_reasoning: bool,
        initial_states_flattened: Vec<f64>,
        time_limit: u64,
    ) -> PyResult<Vec<Node>> {
        let (_, result) = broccoli::run_solver(
            depth,
            num_nodes,
            num_simulation_iterations,
            &predicate_increments,
            use_predicate_reasoning,
            Vec::new(),
            &initial_states_flattened,
            &mut env,
            Duration::new(time_limit, 0),
        );
        result.decision_tree.map(|dt| dt.get_nodes().clone()).ok_or(
            pyo3::exceptions::PyRuntimeError::new_err("Solver did not find a decision tree"),
        )
    }
}

#[derive(FromPyObject)]
struct PythonBasedEnvironment<'py> {
    apply_action: Bound<'py, PyAny>,
    observe_state: Bound<'py, PyAny>,
    get_reward: Bound<'py, PyAny>,
    is_at_terminal_state: Bound<'py, PyAny>,
    rust_reset: Bound<'py, PyAny>,
    environment_info: EnvironmentInfo,
    minimise: bool,
    rewards: bool,
    name: String,
}

impl Environment for PythonBasedEnvironment<'_> {
    fn name(&self) -> &str {
        &self.name
    }
    fn minimise(&self) -> bool {
        self.minimise
    }
    fn rewards(&self) -> bool {
        self.rewards
    }
    fn apply_action(&mut self, action: usize) {
        self.apply_action
            .call1((action,))
            .expect("apply_action should succeed");
    }

    fn observe_state(&self) -> Vec<f64> {
        self.observe_state
            .call0()
            .expect("observe_state should succeed")
            .extract()
            .expect("observe_state should return a list of floats")
    }

    fn get_reward(&self) -> f64 {
        self.get_reward
            .call0()
            .expect("get_reward should succeed")
            .extract()
            .expect("get_reward should return a float")
    }

    fn is_at_terminal_state(&self) -> bool {
        self.is_at_terminal_state
            .call0()
            .expect("is_at_terminal_state should succeed")
            .extract()
            .expect("is_at_terminal_state should return a bool")
    }

    fn reset(&mut self, initial_state: &[f64]) {
        let initial_state = PyArray1::from_slice(self.rust_reset.py(), initial_state);
        self.rust_reset
            .call1((initial_state,))
            .expect("rust_reset should succeed");
    }

    fn environment_info(&self) -> &EnvironmentInfo {
        &self.environment_info
    }
}
