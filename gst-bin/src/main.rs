use gst::GeneralizedSuffixTree;

use anyhow::{bail, Result};

fn main() -> Result<()> {
    let inputs: Vec<String> = std::env::args().skip(1).collect();
    if inputs.len() != 1 {
        bail!("Must provide exactly 1 argument (filename)");
    }
    for input in inputs {
        let data: Vec<_> = std::fs::read_to_string(input)?.chars().collect();
        let st = GeneralizedSuffixTree::from([data]);
        println!("{st}");
    }
    Ok(())
}
