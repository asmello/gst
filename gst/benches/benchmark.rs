use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gst::GeneralizedSuffixTree;

fn criterion_benchmark(c: &mut Criterion) {
    let inputs = [
        include_str!("resources/small.txt"),
        include_str!("resources/medium.txt"),
    ];
    let ids = ["small", "medium"];

    let mut group = c.benchmark_group("from_files");
    for (id, input) in ids.into_iter().zip(inputs.into_iter()) {
        group.throughput(criterion::Throughput::Bytes(input.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(id), &input, |b, &input| {
            b.iter(|| {
                let mut st = GeneralizedSuffixTree::new();
                st.insert(input.chars());
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
