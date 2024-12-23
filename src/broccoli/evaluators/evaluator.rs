use crate::broccoli::{
    environments::environment::EnvironmentInfo, trees::decision_tree::DecisionTree,
};

pub trait Evaluator {
    fn environment_info(&self) -> EnvironmentInfo;
    fn evaluate(&mut self, decision_tree: DecisionTree) -> (DecisionTree, Result<f64, ()>);
    fn register_new_best_score(&mut self, new_best_score: f64);
    fn num_environment_calls(&self) -> usize;
}
