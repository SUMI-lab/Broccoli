use crate::broccoli::{
    broccoli_helper_functions::broccoli_is_integer, environments::environment::Interval,
    trees::simple_tree_from_template_enumerator::SimpleNode,
};

use super::{
    decision_tree::{DecisionTree, Node},
    simple_tree_enumerator::SimpleTreeEnumerator,
};

pub struct DecisionTreeEnumerator {
    increment_steps: Vec<f64>,
    offsets: Vec<f64>,
    simple_tree_enumerator: SimpleTreeEnumerator,
    num_trees_generated: usize,
}

impl DecisionTreeEnumerator {
    pub fn new(
        max_depth: u32,
        max_num_predicate_nodes: u32,
        increment_steps: Vec<f64>,
        state_ranges: Vec<Interval>,
        num_actions: u32,
        use_next_increment_suggestions: bool,
    ) -> Self {
        let max_num_increments = state_ranges
            .iter()
            .enumerate()
            .map(|p| compute_num_increments(p.1.length(), increment_steps[p.0]))
            .collect();

        let simple_tree_enumerator = SimpleTreeEnumerator::new(
            max_depth,
            max_num_predicate_nodes,
            num_actions,
            max_num_increments,
            use_next_increment_suggestions,
        );

        let offsets = state_ranges.iter().map(|interval| interval.min).collect();

        DecisionTreeEnumerator {
            increment_steps,
            offsets,
            simple_tree_enumerator,
            num_trees_generated: 0,
        }
    }

    pub fn next_tree(&mut self) -> Result<DecisionTree, ()> {
        let tree = self.simple_tree_enumerator.next_tree()?;
        self.num_trees_generated += 1;

        //convert the simple tree into a decision tree
        let nodes = tree
            .iter()
            .map(|n| match *n {
                SimpleNode::Unassigned => Node::Null,
                SimpleNode::Leaf { action } => Node::Leaf {
                    action: action as usize,
                },
                SimpleNode::Predicate {
                    feature_id,
                    num_increments,
                    ..
                } => Node::Predicate {
                    feature_id: feature_id as usize,
                    threshold: self.offsets[feature_id as usize]
                        + (num_increments as f64) * self.increment_steps[feature_id as usize],
                },
                SimpleNode::Invalid => unreachable!(),
            })
            .collect();
        Ok(DecisionTree::new(nodes))
    }

    pub fn apply_threshold_reasoning(&mut self, decision_tree: &DecisionTree) {
        //convert thresholds into increments, and pass down
        let next_increment_suggestions: Vec<Option<u32>> = decision_tree
            .get_threshold_distances()
            .iter()
            .enumerate()
            .map(|p| {
                if let Some(threshold_distance) = p.1 {
                    let (feature_id, threshold) = if let Node::Predicate {
                        feature_id,
                        threshold,
                        ..
                    } = decision_tree.get_nodes()[p.0]
                    {
                        (feature_id, threshold)
                    } else {
                        unreachable!();
                    };

                    //some issues arise due to numerical instabilities with floating point values
                    //  in particular it could be that the threshold_distance is slightly smaller than the threshold
                    //      which leads to an increment value that does not change the predicate!
                    //  to avoid this problem, we compute two increments, and take the larger one
                    //  a somewhat better approach would be to look at the current increment of the predicate
                    //      and increase it by one in the worst case

                    let next_increment1 = compute_next_increment(
                        threshold,
                        self.increment_steps[feature_id],
                        self.offsets[feature_id],
                    );

                    let next_increment2 = compute_next_increment(
                        *threshold_distance,
                        self.increment_steps[feature_id],
                        self.offsets[feature_id],
                    );

                    let final_increment = next_increment1.max(next_increment2);

                    Some(final_increment)
                } else {
                    None
                }
            })
            .collect();
        self.simple_tree_enumerator
            .apply_next_increment_suggestions(&next_increment_suggestions);
    }

    pub fn num_trees_generated(&self) -> usize {
        self.num_trees_generated
    }
}

fn compute_num_increments(interval_length: f64, increment_value: f64) -> u32 {
    assert!(interval_length > increment_value);

    //special care needs to be exerted because floating point arithmetic is not precise
    //the problem being solved can be formulated as follows:
    //  how many times can we multiply the increment value while still remaining within the open interval?
    let k = interval_length / increment_value;
    if broccoli_is_integer(k) {
        k.round() as u32 - 1
    } else {
        k.floor() as u32
    }
}

fn compute_next_increment(threshold_distance: f64, increment_value: f64, offset: f64) -> u32 {
    assert!(threshold_distance >= offset);
    let k = (threshold_distance - offset) / increment_value;
    if broccoli_is_integer(k) {
        k.round() as u32 + 1
    } else {
        k.ceil() as u32
    }
}

#[cfg(test)]
mod tests {
    use crate::broccoli::trees::decision_tree_enumerator::{
        compute_next_increment, compute_num_increments,
    };

    #[test]
    fn compute_increments1() {
        assert_eq!(compute_num_increments(5.0, 1.0), 4);
        assert_eq!(compute_num_increments(5.5, 1.0), 5);
        assert_eq!(compute_num_increments(5.99, 1.0), 5);
        assert_eq!(compute_num_increments(6.01, 1.0), 6);
    }

    #[test]
    fn compute_increments2() {
        assert_eq!(compute_num_increments(0.299999999999999, 0.1), 2);
        assert_eq!(compute_num_increments(0.3000000000001, 0.1), 2);
    }

    #[test]
    fn compute_next_increment_suggestions() {
        assert_eq!(compute_next_increment(1.5, 1.0, 0.0), 2);
        assert_eq!(compute_next_increment(1.99, 1.0, 0.0), 2);
        assert_eq!(compute_next_increment(2.0, 1.0, 0.0), 3);
        assert_eq!(compute_next_increment(1.0, 1.0, 0.0), 2);
    }
}
