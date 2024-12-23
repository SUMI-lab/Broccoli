use std::{cmp, fmt};

use crate::broccoli::broccoli_helper_functions::{left_child_id, parent_id, right_child_id};

use super::template_enumerator::TemplateNode;

#[derive(Clone, PartialEq, Eq)]
pub enum SimpleNode {
    Unassigned,
    Leaf {
        action: u32,
    },
    Predicate {
        feature_id: u32,
        num_increments: u32,
        next_increment_suggestion: Option<u32>,
    },
    Invalid,
}

impl SimpleNode {
    pub fn is_leaf(&self) -> bool {
        matches!(self, SimpleNode::Leaf { .. })
    }

    pub fn is_unassigned(&self) -> bool {
        matches!(self, SimpleNode::Unassigned)
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, SimpleNode::Invalid)
    }
}

impl fmt::Display for SimpleNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SimpleNode::Unassigned => write!(f, "(Ø)"),
            SimpleNode::Leaf { action } => write!(f, "(a: {})", action),
            SimpleNode::Predicate {
                feature_id,
                num_increments,
                ..
            } => write!(f, "(f{}: {})", feature_id, num_increments),
            SimpleNode::Invalid => write!(f, "(invalid!)"),
        }
    }
}

impl fmt::Debug for SimpleNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SimpleNode::Unassigned => write!(f, "(Ø)"),
            SimpleNode::Leaf { action } => write!(f, "(a: {})", action),
            SimpleNode::Predicate {
                feature_id,
                num_increments,
                ..
            } => write!(f, "(f{}: {})", feature_id, num_increments),
            SimpleNode::Invalid => write!(f, "(invalid!)"),
        }
    }
}

pub struct SimpleTreeFromTemplateEnumerator {
    state: Vec<SimpleNode>,
    template: Vec<TemplateNode>,
    trail: Vec<u32>,
    is_invalid: bool,
    num_actions: u32,
    max_num_increments: Vec<u32>,
    use_next_increment_suggestions: bool,
}

impl SimpleTreeFromTemplateEnumerator {
    pub fn new(
        template: Vec<TemplateNode>,
        num_actions: u32,
        max_num_increments: Vec<u32>,
        use_next_increment_suggestions: bool,
    ) -> SimpleTreeFromTemplateEnumerator {
        assert!(num_actions >= 2);

        let state: Vec<SimpleNode> = vec![SimpleNode::Unassigned; template.len()];

        SimpleTreeFromTemplateEnumerator {
            state,
            template,
            trail: vec![],
            is_invalid: false,
            num_actions,
            max_num_increments,
            use_next_increment_suggestions,
        }
    }

    pub fn next_tree(&mut self) -> Result<Vec<SimpleNode>, ()> {
        if self.is_invalid {
            return Err(());
        }

        if !self.trail.is_empty() {
            self.backtrack()?;
        }

        loop {
            if self.check_constraints().is_err() {
                self.backtrack()?
            } else {
                let next_node = match self.next_node() {
                    None => {
                        let tree = self.state.clone();
                        return Ok(tree);
                    }
                    Some(next_node) => next_node,
                };

                self.assign_node(next_node);
            }
        }
    }

    fn next_node(&self) -> Option<u32> {
        //we assign leaf nodes before predicate nodes
        //  check to see if there are leaf nodes unassigned
        for p in self.template.iter().enumerate() {
            if let TemplateNode::Leaf = p.1 {
                if let SimpleNode::Unassigned = self.state[p.0] {
                    return Some(p.0 as u32);
                }
            }
        }
        //otherwise select a predicate node
        self.next_predicate_node(0)
    }

    fn next_predicate_node(&self, node_id: u32) -> Option<u32> {
        //traverse the subtrees on the left before the subtrees on the right
        match self.template[node_id as usize] {
            TemplateNode::Leaf => None,
            TemplateNode::Predicate => match self.state[node_id as usize] {
                SimpleNode::Unassigned => Some(node_id),
                SimpleNode::Predicate { .. } => {
                    let left = self.next_predicate_node(left_child_id(node_id as usize) as u32);
                    if left.is_some() {
                        left
                    } else {
                        let right =
                            self.next_predicate_node(right_child_id(node_id as usize) as u32);
                        if right.is_some() {
                            right
                        } else {
                            None
                        }
                    }
                }
                SimpleNode::Leaf { action: _ } => unreachable!(),
                SimpleNode::Invalid => unreachable!(),
            },
            TemplateNode::Unassigned => unreachable!(),
        }
    }

    fn assign_node(&mut self, next_node: u32) {
        match self.template[next_node as usize] {
            TemplateNode::Predicate => match self.state[next_node as usize] {
                SimpleNode::Unassigned => {
                    self.trail.push(next_node);
                    let (lower_bound, _) = self.compute_increment_bounds(next_node, 0);

                    self.state[next_node as usize] = SimpleNode::Predicate {
                        feature_id: 0,
                        num_increments: lower_bound,
                        next_increment_suggestion: None,
                    }
                }
                SimpleNode::Predicate {
                    feature_id,
                    num_increments,
                    next_increment_suggestion,
                } => {
                    //can we go to the next threshold value?
                    let (_, upper_bound) = self.compute_increment_bounds(next_node, feature_id);

                    let next_increment = self
                        .compute_next_increment(num_increments, next_increment_suggestion)
                        .unwrap_or(u32::MAX);

                    if next_increment <= upper_bound {
                        self.state[next_node as usize] = SimpleNode::Predicate {
                            feature_id,
                            num_increments: next_increment,
                            next_increment_suggestion: None,
                        };
                    }
                    //can we go to the next feature?
                    else if feature_id + 1 < self.num_features() {
                        let (lower_bound, _) =
                            self.compute_increment_bounds(next_node, feature_id + 1);

                        self.state[next_node as usize] = SimpleNode::Predicate {
                            feature_id: feature_id + 1,
                            num_increments: lower_bound,
                            next_increment_suggestion: None,
                        };
                    }
                    //otherwise we exhausted all options for this node
                    else {
                        self.state[next_node as usize] = SimpleNode::Invalid;
                    }
                }
                SimpleNode::Leaf { action: _ } => unreachable!(),
                SimpleNode::Invalid => unreachable!(),
            },
            TemplateNode::Leaf => {
                match self.state[next_node as usize] {
                    SimpleNode::Unassigned => {
                        self.trail.push(next_node);
                        self.state[next_node as usize] = SimpleNode::Leaf { action: 0 };
                    }
                    SimpleNode::Leaf { action } => {
                        //can we go to the next action?
                        if action + 1 < self.num_actions {
                            self.state[next_node as usize] =
                                SimpleNode::Leaf { action: action + 1 };
                        } else {
                            self.state[next_node as usize] = SimpleNode::Invalid;
                        }
                    }
                    SimpleNode::Predicate { .. } => unreachable!(),
                    SimpleNode::Invalid => unreachable!(),
                };
            }
            TemplateNode::Unassigned => unreachable!(),
        }
    }

    fn num_features(&self) -> u32 {
        self.max_num_increments.len() as u32
    }

    fn compute_next_increment(
        &self,
        current_increment: u32,
        next_increment_suggestion: Option<u32>,
    ) -> Option<u32> {
        if self.use_next_increment_suggestions {
            next_increment_suggestion
        } else {
            Some(current_increment + 1)
        }
    }

    fn backtrack(&mut self) -> Result<(), ()> {
        //if the trail is empty, conflict
        //if there are invalid nodes at the end, pop them, and then backtrack again
        //  actually there can only be at most one invalid node, and it must be at the end. I think?
        //no invalid nodes
        //take the last node from the trail
        //  assign the next value to the node

        let last_node = match self.trail.last() {
            Some(last_node) => *last_node,
            None => {
                self.is_invalid = true;
                return Err(());
            }
        };

        if self.state[last_node as usize].is_invalid() {
            self.trail.pop();
            self.state[last_node as usize] = SimpleNode::Unassigned;
            return self.backtrack();
        }

        self.assign_node(last_node);

        Ok(())
    }

    fn check_constraints(&mut self) -> Result<(), ()> {
        //if the last node is invalid, the tree is not ok
        //  note that only the last node can be invalid, the other nodes should be non-invalid
        if let Some(last_node) = self.trail.last() {
            if self.state[*last_node as usize].is_invalid() {
                return Err(());
            }
        }

        for node in self.state.iter().enumerate() {
            if let SimpleNode::Predicate {
                feature_id,
                num_increments,
                ..
            } = self.state[node.0]
            {
                let (lower_bound, upper_bound) =
                    self.compute_increment_bounds(node.0 as u32, feature_id);

                if lower_bound > upper_bound {
                    return Err(());
                }

                assert!(num_increments <= upper_bound);
            }
        }

        if self.contains_redundancy(0) {
            return Err(());
        }

        Ok(())
    }

    fn contains_redundancy(&self, node_id: usize) -> bool {
        if self.state[node_id].is_invalid()
            || self.state[node_id].is_leaf()
            || self.state[node_id].is_unassigned()
        {
            false
        } else {
            self.contains_symmetry(left_child_id(node_id), right_child_id(node_id))
                || self.contains_redundancy(left_child_id(node_id))
                || self.contains_redundancy(right_child_id(node_id))
        }
    }

    fn contains_symmetry(&self, node1: usize, node2: usize) -> bool {
        if self.state[node1].is_unassigned()
            || self.state[node2].is_unassigned()
            || self.state[node1].is_invalid()
            || self.state[node2].is_invalid()
        //|| self.state[node1] != self.state[node2]
        {
            false
        } else if self.state[node1] == self.state[node2] {
            //in this branch the nodes are equal, so no need to test whether node2 is a leaf
            if self.state[node1].is_leaf() {
                true
            } else {
                //the nodes are predicate nodes that are equal
                //  so we now check recursively their left and right side
                self.contains_symmetry(left_child_id(node1), left_child_id(node2))
                    || self.contains_symmetry(right_child_id(node1), right_child_id(node2))
            }
        }
        //the two nodes are not equal, so they are not symmetric
        else {
            false
        }
    }

    fn compute_increment_bounds(&self, node_id: u32, feature_id: u32) -> (u32, u32) {
        let (mut lower_bound, mut upper_bound) =
            (1_u32, self.max_num_increments[feature_id as usize]);
        let path = self.get_feature_path(node_id, feature_id);
        for p in path {
            match p.1 {
                true => lower_bound = cmp::max(lower_bound, p.0 + 1),
                false => upper_bound = cmp::min(upper_bound, p.0 - 1),
            }
        }
        (lower_bound, upper_bound)
    }

    fn get_feature_path(&self, mut node_id: u32, feature_id: u32) -> Vec<(u32, bool)> {
        let mut feature_path: Vec<(u32, bool)> = vec![];
        while node_id > 0 {
            let parent_id = parent_id(node_id as usize);
            if let SimpleNode::Predicate {
                feature_id: predicate_feature,
                num_increments,
                ..
            } = self.state[parent_id]
            {
                if feature_id == predicate_feature {
                    let predicate_satisfied = (node_id % 2) == 1; //the left child means the predicate is satisfied
                    feature_path.push((num_increments, predicate_satisfied));
                }
            }
            node_id = parent_id as u32;
        }
        feature_path
    }

    pub fn apply_next_increment_suggestions(&mut self, next_increment_suggestions: &[Option<u32>]) {
        assert!(self.use_next_increment_suggestions);
        assert!(self.state.len() == next_increment_suggestions.len());

        #[allow(clippy::needless_range_loop)]
        for i in 0..self.state.len() {
            if let SimpleNode::Predicate {
                feature_id,
                num_increments,
                next_increment_suggestion,
            } = self.state[i]
            {
                if let Some(new_suggestion) = next_increment_suggestions[i] {
                    match next_increment_suggestion {
                        Some(current_suggestion) => {
                            self.state[i] = SimpleNode::Predicate {
                                feature_id,
                                num_increments,
                                next_increment_suggestion: Some(
                                    current_suggestion.min(new_suggestion),
                                ),
                            }
                        }
                        None => {
                            self.state[i] = SimpleNode::Predicate {
                                feature_id,
                                num_increments,
                                next_increment_suggestion: Some(new_suggestion),
                            }
                        }
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::broccoli::trees::template_enumerator::TemplateNode;

    use super::SimpleTreeFromTemplateEnumerator;

    #[test]
    fn trivial() {
        let mut enumerator =
            SimpleTreeFromTemplateEnumerator::new(vec![TemplateNode::Leaf], 2, vec![], false);

        let result = enumerator.next_tree();
        assert!(result.is_ok());

        let result = enumerator.next_tree();
        assert!(result.is_ok());

        let result = enumerator.next_tree();
        assert!(result.is_err());
    }

    #[test]
    fn depth1() {
        let mut enumerator = SimpleTreeFromTemplateEnumerator::new(
            vec![
                TemplateNode::Predicate,
                TemplateNode::Leaf,
                TemplateNode::Leaf,
            ],
            2,
            vec![2, 3],
            false,
        );

        assert!(enumerator.next_tree().is_ok());
        assert!(enumerator.next_tree().is_ok());
        assert!(enumerator.next_tree().is_ok());
        assert!(enumerator.next_tree().is_ok());
        assert!(enumerator.next_tree().is_ok());

        assert!(enumerator.next_tree().is_ok());
        assert!(enumerator.next_tree().is_ok());
        assert!(enumerator.next_tree().is_ok());
        assert!(enumerator.next_tree().is_ok());
        assert!(enumerator.next_tree().is_ok());

        assert!(enumerator.next_tree().is_err());
    }
}
