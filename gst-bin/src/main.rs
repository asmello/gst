use clap::{Args, Parser, Subcommand};
use gst::GeneralizedSuffixTree;
use std::{
    fs::File,
    io::{stdin, BufRead, BufReader, Error},
};

/// Algorithms based on the Generalized Suffix Tree
#[derive(Parser)]
#[command(author = "André Sá de Mello <asmello@pm.me>")]
#[command(version = "0.1.0")]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(name = "glcs")]
    GeneralizedLongestCommonSubsequence(GeneralizedLongestCommonSubsequenceArgs),
}

#[derive(Args)]
struct GeneralizedLongestCommonSubsequenceArgs {
    /// File to read strings from (if not specified, reads from stdin by default)
    #[arg(short, long)]
    file: Option<String>,

    #[arg(short, long, default_value_t = 2)]
    threshold: usize,

    #[arg(short, long, default_value_t = 2)]
    min_length: usize,
}

fn run_glcp(args: GeneralizedLongestCommonSubsequenceArgs) -> Result<(), Error> {
    let gst = if let Some(file) = args.file.as_deref() {
        let lines = BufReader::new(File::open(file)?)
            .lines()
            .map(|res| res.expect("IO error").chars().collect::<Vec<char>>());
        GeneralizedSuffixTree::from_iter(lines)
    } else {
        let lines = stdin()
            .lock()
            .lines()
            .map(|res| res.expect("IO error").chars().collect::<Vec<char>>());
        GeneralizedSuffixTree::from_iter(lines)
    };

    for common in gst.find_common(args.threshold, args.min_length) {
        println!("{}", String::from_iter(common));
    }

    Ok(())
}

fn main() -> Result<(), Error> {
    let args = Cli::parse();

    match args.command {
        Command::GeneralizedLongestCommonSubsequence(sub_args) => run_glcp(sub_args),
    }
}
