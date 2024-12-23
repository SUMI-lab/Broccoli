use std::{
    cmp,
    fmt::{self},
};

use crate::broccoli::broccoli_helper_functions::{left_child_id, parent_id, right_child_id};

#[derive(Clone)]
pub enum TemplateNode {
    Predicate,
    Leaf,
    Unassigned,
}

impl fmt::Display for TemplateNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TemplateNode::Predicate => write!(f, "[P]"),
            TemplateNode::Leaf => write!(f, "[L]"),
            TemplateNode::Unassigned => write!(f, "[Ø]"),
        }
    }
}

impl fmt::Debug for TemplateNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TemplateNode::Predicate => write!(f, "[P]"),
            TemplateNode::Leaf => write!(f, "[L]"),
            TemplateNode::Unassigned => write!(f, "[Ø]"),
        }
    }
}

pub struct TemplateEnumerator {
    state: Vec<TemplateNode>,
    target_depth: u32,
    target_num_predicate_nodes: u32,
    num_predicate_nodes_assigned: u32,
    trail: Vec<u32>,
    is_invalid: bool,
}

impl TemplateEnumerator {
    pub fn new(depth: u32, num_predicate_nodes: u32) -> TemplateEnumerator {
        let num_nodes: usize = 2_usize.pow(depth + 1) - 1;
        TemplateEnumerator {
            state: vec![TemplateNode::Unassigned; num_nodes],
            target_depth: depth,
            target_num_predicate_nodes: num_predicate_nodes,
            num_predicate_nodes_assigned: 0,
            trail: vec![],
            is_invalid: false,
        }
    }

    pub fn next_template(&mut self) -> Result<Vec<TemplateNode>, ()> {
        if self.is_invalid {
            return Err(());
        }

        loop {
            if self.check_constraints().is_err() {
                self.backtrack()?
            } else {
                let next_node = match self.next_node() {
                    None => {
                        let template = self.state.clone();
                        //backtrack to prepare for the next call of this 'next_template' function
                        let _ = self.backtrack();
                        return Ok(template);
                    }
                    Some(next_node) => next_node,
                };

                self.assign_node(next_node);
            }
        }
    }

    //returns the next node to assign if there is at least one node left unassigned
    //if all nodes are assigned, then returns None
    fn next_node(&self) -> Option<u32> {
        self.next_node_helper(0)
    }

    fn next_node_helper(&self, node_id: u32) -> Option<u32> {
        //traverse the subtrees on the left before the subtrees on the right
        match self.state[node_id as usize] {
            TemplateNode::Predicate => {
                let left = self.next_node_helper(left_child_id(node_id as usize) as u32);
                if left.is_some() {
                    left
                } else {
                    let right = self.next_node_helper(right_child_id(node_id as usize) as u32);
                    if right.is_some() {
                        right
                    } else {
                        None
                    }
                }
            }
            TemplateNode::Leaf => None,
            TemplateNode::Unassigned => Some(node_id),
        }
    }

    fn assign_node(&mut self, next_node: u32) {
        self.state[next_node as usize] = if self.can_be_predicate_node(next_node) {
            self.num_predicate_nodes_assigned += 1;
            TemplateNode::Predicate
        } else {
            TemplateNode::Leaf
        };
        self.trail.push(next_node);
    }

    fn check_constraints(&self) -> Result<(), ()> {
        self.check_depth_constraint()?;
        self.check_num_predicate_nodes_constraint()?;
        Ok(())
    }

    fn check_depth_constraint(&self) -> Result<(), ()> {
        //desired depth cannot be reached or desired depth must be exceeded
        if self.compute_maximum_depth(0) < self.target_depth
            || self.compute_minimum_depth(0, self.remaining_predicate_node_budget())
                > self.target_depth
        {
            Err(())
        } else {
            Ok(())
        }
    }

    fn compute_maximum_depth(&self, node_id: u32) -> u32 {
        match self.state[node_id as usize] {
            TemplateNode::Predicate => {
                let left = self.compute_maximum_depth(left_child_id(node_id as usize) as u32);
                let right = self.compute_maximum_depth(right_child_id(node_id as usize) as u32);
                cmp::max(left, right) + 1
            }
            TemplateNode::Leaf => 0,
            TemplateNode::Unassigned => self.remaining_predicate_node_budget(),
        }
    }

    fn remaining_predicate_node_budget(&self) -> u32 {
        self.target_num_predicate_nodes - self.num_predicate_nodes_assigned
    }

    fn compute_minimum_depth(&self, node_id: u32, num_bonus_predicates: u32) -> u32 {
        match self.state[node_id as usize] {
            TemplateNode::Predicate => {
                //need to consider all possible ways of distributing predicates to the left and right branch
                //  this could be made more efficient by allowing early termination, but for now we ignore that
                let mut min_depth = u32::MAX;
                for left_budget in 0..=num_bonus_predicates {
                    let right_budget = num_bonus_predicates - left_budget;

                    let left_min_depth = self
                        .compute_minimum_depth(left_child_id(node_id as usize) as u32, left_budget);
                    let right_min_depth = self.compute_minimum_depth(
                        right_child_id(node_id as usize) as u32,
                        right_budget,
                    );

                    let new_depth = cmp::max(left_min_depth, right_min_depth) + 1; //+1 because node_id is a predicate node
                    min_depth = cmp::min(min_depth, new_depth);
                }
                min_depth
            }
            TemplateNode::Leaf => 0,
            TemplateNode::Unassigned => {
                //compute the height for the full tree that uses num_bonus_predicates
                (num_bonus_predicates + 1)
                    .checked_next_power_of_two()
                    .unwrap()
                    .ilog2()
            }
        }
    }

    fn check_num_predicate_nodes_constraint(&self) -> Result<(), ()> {
        //assumes we never assign more predicates than possible
        match self.next_node() {
            None => {
                if self.remaining_predicate_node_budget() == 0 {
                    Ok(())
                } else {
                    Err(())
                }
            }
            Some(_) => Ok(()),
        }
    }

    fn backtrack(&mut self) -> Result<(), ()> {
        loop {
            let last_node = match self.trail.last() {
                Some(last_node) => *last_node,
                None => {
                    self.is_invalid = true;
                    return Err(());
                }
            };

            match self.state[last_node as usize] {
                TemplateNode::Predicate => {
                    self.state[last_node as usize] = TemplateNode::Leaf;
                    self.num_predicate_nodes_assigned -= 1;

                    return Ok(());
                }
                TemplateNode::Leaf => {
                    self.state[last_node as usize] = TemplateNode::Unassigned;
                    self.trail.pop();
                }
                TemplateNode::Unassigned => unreachable!(),
            }
        }
    }

    /*fn unassign_subtree(&mut self, node_id: u32) {
        if node_id < self.state.len() as u32 {
            match self.state[node_id as usize] {
                NodeType::Predicate => {
                    self.state[node_id as usize] = NodeType::Unassigned;
                    self.num_predicate_nodes_assigned -= 1;
                    self.unassign_subtree(left_child_id(node_id as usize) as u32);
                    self.unassign_subtree(right_child_id(node_id as usize) as u32);
                }
                NodeType::Leaf => self.state[node_id as usize] = NodeType::Unassigned,
                NodeType::Unassigned => {} //nothing to do in this case
            }
        }
        panic!("NOT SURE IF THIS IS CORRECT NOR NEEDED");
    }*/

    fn can_be_predicate_node(&self, node_id: u32) -> bool {
        !(self.remaining_predicate_node_budget() == 0
            || self.compute_node_depth(node_id) == self.target_depth + 1)
    }

    fn compute_node_depth(&self, mut node_id: u32) -> u32 {
        let mut depth: u32 = 1;
        while node_id > 0 {
            depth += 1;
            node_id = parent_id(node_id as usize) as u32;
        }
        depth
    }
}

#[cfg(test)]
mod tests {
    use super::TemplateEnumerator;

    #[test]
    fn depth_0() {
        let mut enumerator = TemplateEnumerator::new(0, 0);
        let _ = enumerator.next_template();
        let e = enumerator.next_template();
        assert!(e.is_err());
    }

    #[test]
    fn depth_1() {
        let mut enumerator = TemplateEnumerator::new(1, 1);
        let _ = enumerator.next_template();
        let e = enumerator.next_template();
        assert!(e.is_err());
    }

    #[test]
    fn depth_2_full() {
        let mut enumerator = TemplateEnumerator::new(2, 3);
        let _ = enumerator.next_template();
        let e = enumerator.next_template();
        assert!(e.is_err());
    }

    #[test]
    fn depth_2_not_full() {
        let mut enumerator = TemplateEnumerator::new(2, 2);
        let _ = enumerator.next_template();
        let _ = enumerator.next_template();
        let e = enumerator.next_template();
        assert!(e.is_err());
    }

    #[test]
    fn depth_3_num_nodes_3() {
        let mut enumerator = TemplateEnumerator::new(3, 3);
        let mut counter = 0;
        loop {
            let result = enumerator.next_template();
            match result {
                Ok(_) => {
                    counter += 1;
                }
                Err(_) => break,
            }
        }
        assert!(counter == 4);
    }

    #[test]
    fn depth_3_num_nodes_4() {
        let mut enumerator = TemplateEnumerator::new(3, 4);
        let mut counter = 0;
        loop {
            let result = enumerator.next_template();
            match result {
                Ok(_) => {
                    counter += 1;
                }
                Err(_) => break,
            }
        }
        assert!(counter == 6);
    }

    #[test]
    fn predicate_condition_1() {
        let enumerator = TemplateEnumerator::new(2, 3);
        assert!(!enumerator.can_be_predicate_node(3));
    }

    #[test]
    fn node_depth_computation_1() {
        let enumerator = TemplateEnumerator::new(2, 3);
        assert!(enumerator.compute_node_depth(3) == 3);
    }
}
