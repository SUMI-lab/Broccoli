use std::cmp;

use super::{
    simple_tree_from_template_enumerator::{SimpleNode, SimpleTreeFromTemplateEnumerator},
    template_enumerator::TemplateEnumerator,
};

pub struct SimpleTreeEnumerator {
    max_depth: u32,
    max_num_predicate_nodes: u32,
    current_depth: u32,
    current_num_predicate_nodes: u32,
    template_enumerator: TemplateEnumerator,
    tree_from_template_enumerator: SimpleTreeFromTemplateEnumerator,
    num_actions: u32,
    max_num_increments: Vec<u32>,
    is_invalid: bool,
    use_next_increment_suggestions: bool,
}

impl SimpleTreeEnumerator {
    pub fn new(
        max_depth: u32,
        max_num_predicate_nodes: u32,
        num_actions: u32,
        max_num_increments: Vec<u32>,
        use_next_increment_suggestions: bool,
    ) -> SimpleTreeEnumerator {
        let mut template_enumerator = TemplateEnumerator::new(0, 0);
        let template = template_enumerator.next_template().unwrap();
        let tree_from_template_enumerator = SimpleTreeFromTemplateEnumerator::new(
            template,
            num_actions,
            max_num_increments.clone(),
            use_next_increment_suggestions,
        );

        //clamp the depth if it is too large compared to the number of nodes
        let max_depth = cmp::min(max_depth, max_num_predicate_nodes);

        //clamp the number of predicates if the given number
        //  is too large compared to the depth
        let max_num_predicate_nodes =
            cmp::min(max_num_predicate_nodes, (2_usize.pow(max_depth) - 1) as u32);

        SimpleTreeEnumerator {
            max_depth,
            max_num_predicate_nodes,
            current_depth: 0,
            current_num_predicate_nodes: 0,
            template_enumerator,
            tree_from_template_enumerator,
            num_actions,
            max_num_increments,
            is_invalid: false,
            use_next_increment_suggestions,
        }
    }

    pub fn next_tree(&mut self) -> Result<Vec<SimpleNode>, ()> {
        if self.is_invalid {
            return Err(());
        }

        match self.tree_from_template_enumerator.next_tree() {
            //if there is a new tree ready, return it
            Ok(tree) => Ok(tree),
            //otherwise the current template has been exhausted, go to the next template
            Err(_) => match self.template_enumerator.next_template() {
                //check if the next template exists, and if so, use it
                Ok(template) => {
                    self.tree_from_template_enumerator = SimpleTreeFromTemplateEnumerator::new(
                        template,
                        self.num_actions,
                        self.max_num_increments.clone(),
                        self.use_next_increment_suggestions,
                    );
                    self.next_tree()
                }
                //no more templates left for the current depth and num node configuration
                //go on to the next configuration
                Err(_) => {
                    //can we increase the number of nodes?
                    //  yes?
                    let num_allowed_predicate_nodes = self
                        .max_num_predicate_nodes
                        .min(2_u32.pow(self.current_depth) - 1);

                    if self.current_num_predicate_nodes < num_allowed_predicate_nodes {
                        self.current_num_predicate_nodes += 1;
                        println!(
                            "[d = {}, n = {}]",
                            self.current_depth, self.current_num_predicate_nodes
                        );

                        self.template_enumerator = TemplateEnumerator::new(
                            self.current_depth,
                            self.current_num_predicate_nodes,
                        );

                        let template = self.template_enumerator.next_template().unwrap();

                        self.tree_from_template_enumerator = SimpleTreeFromTemplateEnumerator::new(
                            template,
                            self.num_actions,
                            self.max_num_increments.clone(),
                            self.use_next_increment_suggestions,
                        );
                        self.next_tree()
                    }
                    //  no, go to the next depth
                    else {
                        //can we increase the depth?
                        //  yes?
                        if self.current_depth < self.max_depth {
                            self.current_depth += 1;
                            self.current_num_predicate_nodes = self.current_depth; //no point using less nodes than the depth!

                            println!(
                                "[d = {}, n = {}]",
                                self.current_depth, self.current_num_predicate_nodes
                            );

                            self.template_enumerator = TemplateEnumerator::new(
                                self.current_depth,
                                self.current_num_predicate_nodes,
                            );

                            let template = self.template_enumerator.next_template().unwrap();

                            self.tree_from_template_enumerator =
                                SimpleTreeFromTemplateEnumerator::new(
                                    template,
                                    self.num_actions,
                                    self.max_num_increments.clone(),
                                    self.use_next_increment_suggestions,
                                );
                            self.next_tree()
                        }
                        //  no, all options have been explored
                        else {
                            self.is_invalid = true;
                            Err(())
                        }
                    }
                }
            },
        }
    }

    pub fn apply_next_increment_suggestions(&mut self, next_increment_suggestions: &[Option<u32>]) {
        assert!(self.use_next_increment_suggestions);
        self.tree_from_template_enumerator
            .apply_next_increment_suggestions(next_increment_suggestions)
    }
}

#[cfg(test)]
mod tests {
    /*
       pub fn new(
       max_depth: u32,
       max_num_predicate_nodes: u32,
       num_actions: u32,
       max_num_increments: Vec<u32>,
    */

    use super::SimpleTreeEnumerator;

    #[test]
    fn depth0() {
        let mut enumerator = SimpleTreeEnumerator::new(0, 0, 2, vec![], false);
        let _ = enumerator.next_tree();
        let _ = enumerator.next_tree();
        assert!(enumerator.next_tree().is_err());
    }

    #[test]
    fn depth1() {
        let mut enumerator = SimpleTreeEnumerator::new(1, 1, 2, vec![2], false);
        //depth 0 case
        let _ = enumerator.next_tree();
        let _ = enumerator.next_tree();
        //depth 1
        //  first increment
        let _ = enumerator.next_tree();
        let _ = enumerator.next_tree();
        //  second increment
        let _ = enumerator.next_tree();
        let _ = enumerator.next_tree();

        //  done
        assert!(enumerator.next_tree().is_err());
    }

    #[test]
    fn depth2() {
        let mut enumerator = SimpleTreeEnumerator::new(2, 3, 2, vec![2], false);

        let mut counter = 0;
        loop {
            match enumerator.next_tree() {
                Ok(tree) => {
                    println!("{}: {:?}", counter, tree);
                    counter += 1;
                }
                Err(_) => break,
            }
        }

        /*//depth 0 case
        let _ = enumerator.next_tree();
        let _ = enumerator.next_tree();
        //depth 1
        //  first increment
        let _ = enumerator.next_tree();
        let _ = enumerator.next_tree();
        //  second increment
        let _ = enumerator.next_tree();
        let _ = enumerator.next_tree();
        //depth 2


        assert!(enumerator.next_tree().is_err());*/
    }
}
