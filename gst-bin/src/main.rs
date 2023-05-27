use gst::SuffixTree;

use anyhow::{bail, Result};

fn main() -> Result<()> {
    let inputs: Vec<String> = std::env::args().skip(1).collect();
    if inputs.len() != 1 {
        bail!("Must provide exactly 1 argument (filename)");
    }
    for input in inputs {
        let data = std::fs::read_to_string(input)?.chars().collect();
        let st = SuffixTree::new(data);
        println!("{st}");
    }
    Ok(())
}
