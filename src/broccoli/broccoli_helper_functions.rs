use std::{cmp::Ordering, path::Path, process::exit};

use plotlib::{
    page::Page,
    repr::Plot,
    style::{LineJoin, LineStyle, PointStyle},
    view::ContinuousView,
};

pub fn brocolli_within_range(n: f64, low: f64, high: f64) -> bool {
    broccoli_greater_or_equal(n, low) && broccoli_greater_or_equal(high, n)
}

//greater or equal for floating point values
pub fn broccoli_greater_or_equal(a: f64, b: f64) -> bool {
    a > b || broccoli_equal(a, b)
}

//equality for floating point values
pub fn broccoli_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < 0.000001
}

//tests whether floating point number is (close) to an integer
pub fn broccoli_is_integer(a: f64) -> bool {
    broccoli_equal(a.fract(), 0.0) || broccoli_equal(a.fract(), 1.0)
}

pub fn left_child_id(node_id: usize) -> usize {
    2 * node_id + 1
}

pub fn right_child_id(node_id: usize) -> usize {
    2 * node_id + 2
}

pub fn parent_id(node_id: usize) -> usize {
    assert!(node_id > 0, "Root node does not have a parent.");
    (node_id - 1) / 2
}

pub fn extract_initial_states(
    initial_states_flattened: &[f64],
    num_state_variables: usize,
) -> Vec<Vec<f64>> {
    if initial_states_flattened.is_empty() {
        println!("Error: no initial state provided.");
        exit(1);
    }

    if initial_states_flattened.len() % num_state_variables != 0 {
        println!("Error: initial states not properly specified.");
        println!(
            "\t{} values provided, but this is not a proper multiple of {} that is expected.",
            initial_states_flattened.len(),
            num_state_variables
        );
        exit(1);
    }

    let initial_states: Vec<_> = initial_states_flattened
        .chunks(num_state_variables)
        .map(|p| p.to_vec())
        .collect();

    println!("Num initial states: {}", initial_states.len());
    for state in &initial_states {
        println!("\t{:?}", state);
    }

    initial_states
}

pub fn check_predicate_increments(predicate_increments: &[f64], num_state_variables: usize) {
    if predicate_increments.len() != num_state_variables {
        println!("Error: predicate increments does not match the number of features.");
        println!("\tProvided increments: {:?}", predicate_increments);
        exit(1);
    }

    if predicate_increments.iter().any(|p| p.is_sign_negative()) {
        println!("Error: predicate increments contain negative values?");
        println!("\tProvided increments: {:?}", predicate_increments);
        exit(1);
    }

    if predicate_increments
        .iter()
        .any(|p| broccoli_greater_or_equal(0.0, *p))
    {
        println!("Error: predicate increments contain very small values?");
        println!("\tProvided increments: {:?}", predicate_increments);
        exit(1);
    }

    println!("Predicate increments:");
    for p in predicate_increments.iter().enumerate() {
        println!("\tf{}: {}", p.0, p.1);
    }
}

fn broccoli_cmp(x: f64, y: f64) -> Ordering {
    if broccoli_equal(x, y) {
        Ordering::Equal
    } else if broccoli_greater_or_equal(x, y) {
        Ordering::Greater
    } else {
        Ordering::Less
    }
}

pub fn broccoli_plot(
    x_values: &[f64],
    x_name: &str,
    y_values: &[f64],
    y_name: &str,
    plot_name: &str,
) {
    assert!(
        x_values.len() == y_values.len(),
        "Error: plotting issue, x and y vectors are not of same size."
    );
    assert!(
        !x_values.is_empty() && !y_values.is_empty(),
        "Error: plotting with empty arrays?"
    );

    let data: Vec<(f64, f64)> = x_values
        .iter()
        .enumerate()
        .map(|p| (*p.1, y_values[p.0]))
        .collect();

    let x_min = *x_values
        .iter()
        .min_by(|&&x, &&y| broccoli_cmp(x, y))
        .unwrap();
    let x_max = *x_values
        .iter()
        .max_by(|&&x, &&y| broccoli_cmp(x, y))
        .unwrap();

    let y_min = *y_values
        .iter()
        .min_by(|&&x, &&y| broccoli_cmp(x, y))
        .unwrap();
    let y_max = *y_values
        .iter()
        .max_by(|&&x, &&y| broccoli_cmp(x, y))
        .unwrap();

    let plot = Plot::new(data)
        .line_style(
            LineStyle::new()
                .colour("burlywood")
                .linejoin(LineJoin::Round),
        )
        .point_style(PointStyle::new());

    let view = ContinuousView::new()
        .add(plot)
        .x_range(x_min, x_max)
        .y_range(y_min, y_max)
        .x_label(x_name)
        .y_label(y_name);

    let path = Path::new("plots").join(format!("{}.svg", plot_name));

    Page::single(&view).save(path).expect("Problem plotting?");
}

#[cfg(test)]
mod tests {
    use crate::broccoli::broccoli_helper_functions::{
        broccoli_equal, broccoli_greater_or_equal, broccoli_is_integer, extract_initial_states,
    };

    #[test]
    fn ge1() {
        assert!(broccoli_greater_or_equal(5.1_f64, 3.0_f64));
    }

    #[test]
    fn ge2() {
        assert!(broccoli_greater_or_equal(5.1_f64, 5.1_f64));
    }

    #[test]
    fn ge3() {
        assert!(broccoli_greater_or_equal(-5.0_f64, -5.11_f64));
    }

    #[test]
    fn ge4() {
        assert!(!broccoli_greater_or_equal(-5.0_f64, 10.0_f64));
    }

    #[test]
    fn ge5() {
        assert!(broccoli_greater_or_equal(-5.0_f64, -10.0_f64));
    }

    #[test]
    fn eq1() {
        assert!(broccoli_equal(-5.0_f64, -5.0_f64));
    }

    #[test]
    fn eq2() {
        assert!(!broccoli_equal(5.0_f64, -5.0_f64));
    }

    #[test]
    fn eq3() {
        assert!(!broccoli_equal(6.0_f64, 6.11_f64));
    }

    #[test]
    fn eq4() {
        assert!(!broccoli_equal(6.0_f64, -6.0_f64));
    }

    #[test]
    fn is_integer1() {
        assert!(broccoli_is_integer(5.0_f64));
    }

    #[test]
    fn is_integer2() {
        assert!(!broccoli_is_integer(5.01_f64));
    }

    #[test]
    fn is_integer3() {
        assert!(!broccoli_is_integer(3.5_f64));
    }

    #[test]
    fn is_integer4() {
        assert!(!broccoli_is_integer(-0.1_f64));
    }

    #[test]
    fn extract_states_1() {
        let values = [0.5, 1.0];
        let initial_states = extract_initial_states(&values, 1);
        assert!(initial_states == [[0.5], [1.0]])
    }

    #[test]
    fn extract_states_2() {
        let values = [0.5, 1.0, 10.0, 20.0];

        let initial_states = extract_initial_states(&values, 2);
        assert!(initial_states == [[0.5, 1.0], [10.0, 20.0]]);

        let comparison: Vec<Vec<f64>> = values.chunks(2).map(|p| p.to_vec()).collect();
        assert!(initial_states == comparison);
    }

    #[test]
    fn extract_states_3() {
        let values = [0.5, 1.0, 10.0, 20.0, 100.0, 0.0];
        let initial_states = extract_initial_states(&values, 2);
        assert!(initial_states == [[0.5, 1.0], [10.0, 20.0], [100.0, 0.0]]);
    }
}
