use std::time::Duration;

use crate::broccoli::trees::decision_tree_enumerator::DecisionTreeEnumerator;

use super::{evaluators::evaluator::Evaluator, trees::decision_tree::DecisionTree};

pub struct Broccoli {}

pub struct BroccoliOutput {
    pub decision_tree: Option<DecisionTree>,
    pub score: Option<f64>,
    pub runtime: Duration,
    pub num_trees_considered: usize,
    pub num_environment_calls: usize,
}

impl BroccoliOutput {
    pub fn print_basic_stats(&self) {
        println!("Runtime: {:.2?}", self.runtime);
        println!(
            "Num explicitly considered trees: {}",
            self.num_trees_considered
        );
        println!("Num environment calls: {}", self.num_environment_calls);

        if let Some(score) = self.score {
            println!("Score: {}", score);
            println!("Tree:\n{}", self.decision_tree.clone().unwrap());
        } else {
            println!("No tree found.");
        }
    }
}

//predicate generator

impl Broccoli {
    pub fn compute_decision_tree(
        decision_tree_depth: u32,
        num_predicate_nodes: u32,
        mut evaluator: Box<dyn Evaluator>,
        predicate_increments: &[f64],
        use_predicate_reasoning: bool,
    ) -> BroccoliOutput {
        assert!(evaluator.environment_info().num_features() == predicate_increments.len());

        println!("Starting broccoli...");

        let mut trees_with_unused_nodes = 0;

        let time_tracker = std::time::Instant::now();
        let mut enumerator = DecisionTreeEnumerator::new(
            decision_tree_depth,
            num_predicate_nodes,
            //2_u32.pow(decision_tree_depth) - 1,
            predicate_increments.to_vec(),
            evaluator.environment_info().ranges(),
            evaluator.environment_info().num_actions() as u32,
            use_predicate_reasoning,
        );

        let mut best_score: Option<f64> = None;
        let mut best_tree: Option<DecisionTree> = None;

        while let Ok(decision_tree) = enumerator.next_tree() {
            let (decision_tree_with_info, new_score) = evaluator.evaluate(decision_tree);

            if let Ok(new_value) = new_score {
                best_score = Some(new_value);
                best_tree = Some(decision_tree_with_info.clone());
                evaluator.register_new_best_score(new_value);
                println!(
                    "...{} [{} nodes] ({:.2?})",
                    new_value,
                    decision_tree_with_info.num_predicate_nodes(),
                    time_tracker.elapsed()
                );
            }

            let mut num_unused_nodes = 0;
            for i in 0..decision_tree_with_info.get_nodes().len() {
                if decision_tree_with_info.get_nodes()[i].is_predicate()
                    && decision_tree_with_info.get_frequencies()[i] == 0
                {
                    num_unused_nodes += 1;
                }
            }

            if num_unused_nodes > 0 {
                trees_with_unused_nodes += 1;
                //println!("un: {}", decision_tree_with_info);
            }

            if use_predicate_reasoning {
                enumerator.apply_threshold_reasoning(&decision_tree_with_info);
            }
        }

        println!("Trees with unused nodes: {}", trees_with_unused_nodes);

        BroccoliOutput {
            decision_tree: best_tree,
            score: best_score,
            runtime: time_tracker.elapsed(),
            num_trees_considered: enumerator.num_trees_generated(),
            num_environment_calls: evaluator.num_environment_calls(),
        }
    }
}
