use std::fmt;

use crate::broccoli::broccoli_helper_functions::{
    broccoli_equal, broccoli_greater_or_equal, left_child_id, right_child_id,
};

#[derive(Clone)]
pub enum Node {
    Null,
    Leaf { action: usize },
    Predicate { feature_id: usize, threshold: f64 },
}

impl Node {
    pub fn is_predicate(&self) -> bool {
        matches!(
            self,
            Node::Predicate {
                feature_id: _,
                threshold: _
            }
        )
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Node::Null)
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        if self.is_null() && other.is_null() {
            return true;
        }

        match self {
            Node::Null => other.is_null(),
            Node::Leaf { action } => match other {
                Node::Null => false,
                Node::Leaf {
                    action: action_other,
                } => action == action_other,
                Node::Predicate { .. } => false,
            },
            Node::Predicate {
                feature_id,
                threshold,
            } => match other {
                Node::Null => false,
                Node::Leaf { .. } => false,
                Node::Predicate {
                    feature_id: feature_other,
                    threshold: threshold_other,
                } => feature_id == feature_other && broccoli_equal(*threshold, *threshold_other),
            },
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Node::Null => {
                write!(f, "Ø")
            }
            Node::Leaf { action } => {
                write!(f, "a: {}", action)
            }
            Node::Predicate {
                feature_id,
                threshold,
            } => {
                write!(f, "[x_{} >= {:.2}]", feature_id, threshold)
            }
        }
    }
}

#[derive(Clone)]
pub struct DecisionTree {
    nodes: Vec<Node>, //[i] is the i-th node. The children of the i-th node are 2*i+1 and 2*i+2. The root is node [0].
    frequencies: Vec<usize>,
    threshold_distance: Vec<Option<f64>>,
}

impl PartialEq for DecisionTree {
    fn eq(&self, other: &Self) -> bool {
        //we are just comparing nodes, and ignoring frequencies and threshold distances
        //  some older version of trees contained unnecessary null nodes, so here we take into account to skip those
        let m = self.nodes.len().min(other.nodes.len());
        for j in 0..m {
            if self.nodes[j] != other.nodes[j] {
                return false;
            }
        }

        for j in m..self.nodes.len() {
            if !self.nodes[j].is_null() {
                return false;
            }
        }

        for j in m..other.nodes.len() {
            if !other.nodes[j].is_null() {
                return false;
            }
        }

        true
    }
}

impl DecisionTree {
    pub fn new(nodes: Vec<Node>) -> Self {
        //todo some checks on the nodes
        let num_nodes = nodes.len();
        DecisionTree {
            nodes,
            frequencies: vec![0; num_nodes],
            threshold_distance: vec![None; num_nodes],
        }
    }

    pub fn get_action(&mut self, state: &Vec<f64>) -> usize {
        self.traverse(0, state)
    }

    fn update_threshold_distance(&mut self, node_id: usize, new_value: f64) {
        match self.threshold_distance[node_id] {
            Some(old_value) => {
                if broccoli_greater_or_equal(old_value, new_value) {
                    self.threshold_distance[node_id] = Some(new_value);
                }
            }
            None => self.threshold_distance[node_id] = Some(new_value),
        }
    }

    fn traverse(&mut self, node_id: usize, state: &Vec<f64>) -> usize {
        self.frequencies[node_id] += 1;

        match self.nodes[node_id] {
            Node::Null => unreachable!(),
            Node::Leaf { action } => action,
            Node::Predicate {
                feature_id,
                threshold,
            } => {
                if broccoli_greater_or_equal(state[feature_id], threshold) {
                    assert!(
                        self.debug_check_threshold_distance_candidate(node_id, state[feature_id])
                    );
                    self.update_threshold_distance(node_id, state[feature_id]);
                    self.traverse(left_child_id(node_id), state)
                } else {
                    self.traverse(right_child_id(node_id), state)
                }
            }
        }
    }

    fn debug_check_threshold_distance_candidate(&self, node_id: usize, new_value: f64) -> bool {
        match self.nodes[node_id] {
            Node::Null => unreachable!(),
            Node::Leaf { action: _ } => unreachable!(),
            Node::Predicate {
                feature_id: _,
                threshold,
            } => {
                self.threshold_distance[node_id].is_none()
                    || broccoli_greater_or_equal(new_value, threshold)
            }
        }
    }

    pub fn get_nodes(&self) -> &Vec<Node> {
        &self.nodes
    }

    pub fn get_frequencies(&self) -> &Vec<usize> {
        &self.frequencies
    }

    pub fn get_threshold_distances(&self) -> &Vec<Option<f64>> {
        &self.threshold_distance
    }

    pub fn merge_threshold_distances(&mut self, source: &DecisionTree) {
        assert!(self.nodes.len() == source.nodes.len());
        //todo check that it is the same tree

        for node_id in 0..self.nodes.len() {
            self.frequencies[node_id] += source.frequencies[node_id];

            if self.nodes[node_id].is_predicate() {
                if let Some(new_value) = source.threshold_distance[node_id] {
                    self.update_threshold_distance(node_id, new_value);
                }
            }
        }
    }

    pub fn num_predicate_nodes(&self) -> usize {
        assert!(self.threshold_distance.len() == self.frequencies.len());
        assert!(self.nodes.len() == self.frequencies.len());
        self.nodes.iter().filter(|node| node.is_predicate()).count()
    }

    //pub fn num_predicate_nodes
}

impl fmt::Display for DecisionTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for node in self.nodes.iter().enumerate() {
            if !matches!(node.1, Node::Null) {
                writeln!(f, "n{}: {} ({})", node.0, node.1, self.frequencies[node.0])?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use crate::broccoli::broccoli_helper_functions::broccoli_equal;

    use super::{DecisionTree, Node};

    fn get_sample_tree1() -> DecisionTree {
        let nodes: Vec<Node> = vec![
            Node::Predicate {
                feature_id: 0,
                threshold: 2.5,
            },
            Node::Predicate {
                feature_id: 1,
                threshold: 10.0,
            },
            Node::Predicate {
                feature_id: 2,
                threshold: 5.0,
            },
            Node::Leaf { action: 0 },
            Node::Leaf { action: 1 },
            Node::Leaf { action: 2 },
            Node::Leaf { action: 3 },
        ];
        DecisionTree::new(nodes)
    }

    #[test]
    fn num_nodes() {
        let tree = get_sample_tree1();
        assert_eq!(tree.nodes.len(), 7);
    }

    #[test]
    fn actions() {
        let mut tree = get_sample_tree1();
        let mut features: Vec<f64> = vec![5.0, 2.0, 10.0];
        assert_eq!(tree.get_action(&features), 1);
        features[1] = 9.99;
        assert_eq!(tree.get_action(&features), 1);
        features[1] = 10.1;
        assert_eq!(tree.get_action(&features), 0);
        features[0] = -2.0;
        assert_eq!(tree.get_action(&features), 2);
        features[2] = -5.1;
        assert_eq!(tree.get_action(&features), 3);
    }

    #[test]
    fn frequencies() {
        let mut tree = get_sample_tree1();
        let features: Vec<f64> = vec![5.0, 2.0, 10.0];
        let _ = tree.get_action(&features);
        assert_eq!(tree.frequencies[0], 1);
        assert_eq!(tree.frequencies[1], 1);
        assert_eq!(tree.frequencies[2], 0);
        assert_eq!(tree.frequencies[3], 0);
        assert_eq!(tree.frequencies[4], 1);
        assert_eq!(tree.frequencies[5], 0);
        assert_eq!(tree.frequencies[6], 0);
        let features = vec![2.5, 10.0, 10.0];
        let _ = tree.get_action(&features);
        assert_eq!(tree.frequencies[0], 2);
        assert_eq!(tree.frequencies[1], 2);
        assert_eq!(tree.frequencies[2], 0);
        assert_eq!(tree.frequencies[3], 1);
        assert_eq!(tree.frequencies[4], 1);
        assert_eq!(tree.frequencies[5], 0);
        assert_eq!(tree.frequencies[6], 0);
    }

    #[test]
    fn overcomes() {
        let mut tree = get_sample_tree1();
        let features: Vec<f64> = vec![5.0, 2.0, 10.0];
        assert!(tree.threshold_distance.iter().all(|p| p.is_none()));
        //2.5, 10, 5
        let _ = tree.get_action(&features);
        assert!(broccoli_equal(tree.threshold_distance[0].unwrap(), 5.0));
        assert!(tree.threshold_distance[1].is_none());
        assert!(tree.threshold_distance[2].is_none());

        let features = vec![6.0, 2.0, 10.0];
        let _ = tree.get_action(&features);
        assert!(broccoli_equal(tree.threshold_distance[0].unwrap(), 5.0));
        assert!(tree.threshold_distance[1].is_none());
        assert!(tree.threshold_distance[2].is_none());

        let features = vec![4.1, 2.0, 10.0];
        let _ = tree.get_action(&features);
        assert!(broccoli_equal(tree.threshold_distance[0].unwrap(), 4.1));
        assert!(tree.threshold_distance[1].is_none());
        assert!(tree.threshold_distance[2].is_none());

        let features = vec![5.0, 15.0, 10.0];
        let _ = tree.get_action(&features);
        assert!(broccoli_equal(tree.threshold_distance[0].unwrap(), 4.1));
        assert!(broccoli_equal(tree.threshold_distance[1].unwrap(), 15.0));
        assert!(tree.threshold_distance[2].is_none());
    }
}
