use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

const NULL: usize = 0;
const ROOT: usize = 1;

#[derive(Default, Debug, PartialEq, Eq)]
struct Node {
    source: usize,
    start: usize,
    end: usize,
}

impl Node {
    fn new(source: usize, start: usize, end: usize) -> Self {
        Self {
            source,
            start,
            end,
            ..Default::default()
        }
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct SuffixTree<E>
where
    E: Copy + Default + Eq + Hash + Debug,
{
    data: Vec<Vec<E>>,
    nodes: Vec<Node>,
    edges: HashMap<usize, HashMap<E, usize>>,
    links: HashMap<usize, usize>,
    terminators: HashMap<usize, Vec<usize>>,
}

impl<E, I, const N: usize> From<[I; N]> for SuffixTree<E>
where
    I: IntoIterator<Item = E>,
    E: Copy + Default + Eq + Hash + Debug,
{
    fn from(arr: [I; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<E, I> FromIterator<I> for SuffixTree<E>
where
    I: IntoIterator<Item = E>,
    E: Copy + Default + Eq + Hash + Debug,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = I>,
    {
        let mut st = Self::new();
        for sequence in iter {
            st.add(sequence.into_iter().collect());
        }
        st
    }
}

impl<E> SuffixTree<E>
where
    E: Copy + Default + Eq + Hash + Debug,
{
    pub fn new() -> Self {
        Self {
            nodes: vec![Node::new(0, 0, 0)],
            links: HashMap::from([(1, 0)]),
            ..Default::default()
        }
    }

    pub fn add(&mut self, data: Vec<E>) {
        self.add_ukkonen(data);
    }

    fn new_node(&mut self, source: usize, start: usize, end: usize) -> usize {
        self.nodes.push(Node::new(source, start, end));
        self.nodes.len()
    }

    fn add_edge(&mut self, src: usize, key: E, dst: usize) {
        self.edges.entry(src).or_default().insert(key, dst);
    }

    fn edge_exists(&self, src: usize, key: E) -> bool {
        match src {
            0 => true,
            n => self
                .edges
                .get(&n)
                .map(|edges| edges.contains_key(&key))
                .unwrap_or_default(),
        }
    }

    fn edge(&self, node: usize, key: E) -> usize {
        match node {
            0 => 1,
            n => self.edges[&n][&key],
        }
    }

    fn node(&self, node: usize) -> &Node {
        &self.nodes[node - 1]
    }

    fn node_mut(&mut self, node: usize) -> &mut Node {
        &mut self.nodes[node - 1]
    }

    fn link(&self, node: usize) -> usize {
        self.links[&node]
    }

    fn get(&self, source: usize, pos: usize) -> E {
        self.data[source][pos - 1]
    }

    fn end(&self, source: usize) -> usize {
        self.data[source].len()
    }

    fn add_link(&mut self, src: usize, dst: usize) {
        self.links.insert(src, dst);
    }

    fn add_terminator(&mut self, node: usize, source: usize) {
        self.terminators.entry(node).or_default().push(source);
    }

    fn add_ukkonen(&mut self, data: Vec<E>) {
        // let z = self.data.len();
        self.data.push(data);

        let mut node = ROOT;
        let mut start = 1;
        for i in 1..=self.end(0) {
            (node, start) = self.update(node, start, i);
            (node, start) = self.canonize(node, start, i);
        }

        self.finalize(node, start);
    }

    fn finalize(&mut self, mut node: usize, mut start: usize) {
        let (mut endpoint, mut curr) = self.split_last(node, start);
        let mut prev = ROOT;
        while !endpoint {
            self.add_terminator(curr, 0);
            if prev != ROOT {
                self.add_link(prev, curr);
            }
            prev = curr;
            (node, start) = self.canonize(self.link(node), start, self.end(0));
            (endpoint, curr) = self.split_last(node, start);
        }
    }

    fn split_last(&mut self, node: usize, start: usize) -> (bool, usize) {
        let end = self.end(0);
        if start <= end {
            let t_start = self.get(0, start);
            let child = self.edge(node, t_start);
            let t_mid = self.get(0, 1 + self.node(child).start + end - start);
            let mid = self.new_node(
                0,
                self.node(child).start,
                self.node(child).start + end - start,
            );
            self.node_mut(child).start += end - start + 1;
            self.add_edge(mid, t_mid, child);
            self.add_edge(node, t_start, mid);
            (false, mid)
        } else {
            if node == NULL {
                (true, node)
            } else {
                (false, node)
            }
        }
    }

    fn update(&mut self, mut node: usize, mut start: usize, i: usize) -> (usize, usize) {
        let t_i = self.get(0, i);
        let mut prev = ROOT;
        let (mut endpoint, mut curr) = self.test_and_split(node, start, i - 1, t_i);
        while !endpoint {
            let new = self.new_node(0, i, self.end(0));
            self.add_edge(curr, t_i, new);
            self.add_terminator(new, 0);
            if prev != ROOT {
                self.add_link(prev, curr);
            }
            prev = curr;
            (node, start) = self.canonize(self.link(node), start, i - 1);
            (endpoint, curr) = self.test_and_split(node, start, i - 1, t_i);
        }
        if prev != ROOT {
            self.add_link(prev, node);
        }
        (node, start)
    }

    fn test_and_split(&mut self, node: usize, start: usize, end: usize, t: E) -> (bool, usize) {
        if start <= end {
            let t_start = self.get(0, start);
            let child = self.edge(node, t_start);
            let t_mid = self.get(0, 1 + self.node(child).start + end - start);
            if t == t_mid {
                (true, node)
            } else {
                let mid = self.new_node(
                    0,
                    self.node(child).start,
                    self.node(child).start + end - start,
                );
                self.node_mut(child).start += end - start + 1;
                self.add_edge(mid, t_mid, child);
                self.add_edge(node, t_start, mid);
                (false, mid)
            }
        } else {
            if self.edge_exists(node, t) {
                (true, node)
            } else {
                (false, node)
            }
        }
    }

    fn canonize(&mut self, mut node: usize, mut start: usize, end: usize) -> (usize, usize) {
        if end >= start {
            let mut t_start = self.get(0, start);
            let mut child = self.edge(node, t_start);
            while end >= start && self.node(child).end - self.node(child).start <= end - start {
                start += self.node(child).end - self.node(child).start + 1;
                node = child;
                if start <= end {
                    t_start = self.get(0, start);
                    child = self.edge(node, t_start);
                }
            }
        }
        (node, start)
    }
}

impl<E> std::fmt::Display for SuffixTree<E>
where
    E: Copy + Default + Eq + Hash + Debug,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "1\n")?;
        if !self.edges.contains_key(&ROOT) {
            return Ok(());
        }
        let n = self.edges[&ROOT].len();
        let mut stack: Vec<(usize, String, bool)> = Vec::new();
        for (idx, &child) in self.edges[&ROOT].values().enumerate() {
            stack.push((child, "".into(), idx == n - 1));
            while let Some((curr, mut prefix, is_last)) = stack.pop() {
                let to_add = if is_last { "└──" } else { "├──" };
                let curr_prefix: String = prefix.chars().chain(to_add.chars()).collect();
                write!(
                    fmt,
                    "{curr_prefix} {curr} {:?}",
                    &self.data[self.node(curr).source]
                        [self.node(curr).start - 1..self.node(curr).end],
                )?;
                if self.terminators.contains_key(&curr) {
                    write!(fmt, "--{:?}", self.terminators[&curr])?;
                }
                if self.links.contains_key(&curr) {
                    write!(fmt, " ➔ {}", self.links[&curr])?;
                }
                write!(fmt, "\n")?;
                if self.edges.contains_key(&curr) {
                    if is_last {
                        prefix.push_str("    ");
                    } else {
                        prefix.push_str("│   ");
                    }
                    for (idx, &child) in self.edges[&curr].values().enumerate() {
                        stack.push((child, prefix.clone(), idx == 0));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    // #[test]
    // fn test() {
    //     let str = "caca";
    //     let result = SuffixTree::from([str.to_owned().chars()]);
    //     println!("{result}\n{result:#?}");
    // }

    #[test]
    fn test_aabccb_unique() {
        let str = "aabccb$";
        let expected = SuffixTree {
            data: vec![vec!['a', 'a', 'b', 'c', 'c', 'b', '$']],
            nodes: vec![
                Node {
                    source: 0,
                    start: 0,
                    end: 0,
                },
                Node {
                    source: 0,
                    start: 2,
                    end: 7,
                },
                Node {
                    source: 0,
                    start: 1,
                    end: 1,
                },
                Node {
                    source: 0,
                    start: 3,
                    end: 7,
                },
                Node {
                    source: 0,
                    start: 4,
                    end: 7,
                },
                Node {
                    source: 0,
                    start: 5,
                    end: 7,
                },
                Node {
                    source: 0,
                    start: 4,
                    end: 4,
                },
                Node {
                    source: 0,
                    start: 6,
                    end: 7,
                },
                Node {
                    source: 0,
                    start: 3,
                    end: 3,
                },
                Node {
                    source: 0,
                    start: 7,
                    end: 7,
                },
                Node {
                    source: 0,
                    start: 7,
                    end: 7,
                },
            ],
            edges: HashMap::from([
                (1, HashMap::from([('a', 3), ('b', 9), ('c', 7), ('$', 11)])),
                (3, HashMap::from([('a', 2), ('b', 4)])),
                (7, HashMap::from([('b', 8), ('c', 6)])),
                (9, HashMap::from([('c', 5), ('$', 10)])),
            ]),
            links: HashMap::from([(1, 0), (3, 1), (7, 1), (9, 1)]),
            terminators: HashMap::from([
                (1, vec![0]),
                (2, vec![0]),
                (4, vec![0]),
                (5, vec![0]),
                (6, vec![0]),
                (8, vec![0]),
                (10, vec![0]),
                (11, vec![0]),
            ]),
        };
        let result = SuffixTree::from([str.to_owned().chars()]);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_aabccb() {
        let str = "aabccb";
        let expected = SuffixTree {
            data: vec![vec!['a', 'a', 'b', 'c', 'c', 'b']],
            nodes: vec![
                Node {
                    source: 0,
                    start: 0,
                    end: 0,
                },
                Node {
                    source: 0,
                    start: 2,
                    end: 6,
                },
                Node {
                    source: 0,
                    start: 1,
                    end: 1,
                },
                Node {
                    source: 0,
                    start: 3,
                    end: 6,
                },
                Node {
                    source: 0,
                    start: 4,
                    end: 6,
                },
                Node {
                    source: 0,
                    start: 5,
                    end: 6,
                },
                Node {
                    source: 0,
                    start: 4,
                    end: 4,
                },
                Node {
                    source: 0,
                    start: 6,
                    end: 6,
                },
                Node {
                    source: 0,
                    start: 3,
                    end: 3,
                },
            ],
            edges: HashMap::from([
                (1, HashMap::from([('a', 3), ('b', 9), ('c', 7)])),
                (3, HashMap::from([('a', 2), ('b', 4)])),
                (7, HashMap::from([('c', 6), ('b', 8)])),
                (9, HashMap::from([('c', 5)])),
            ]),
            links: HashMap::from([(1, 0), (3, 1), (7, 1), (9, 1)]),
            terminators: HashMap::from([
                (1, vec![0]),
                (2, vec![0]),
                (4, vec![0]),
                (5, vec![0]),
                (6, vec![0]),
                (8, vec![0]),
                (9, vec![0]),
            ]),
        };
        let result = SuffixTree::from([str.to_owned().chars()]);
        assert_eq!(expected, result);
    }

    // #[test]
    // fn test_multiple() {
    //     let expected = SuffixTree {
    //         data: vec![vec!['A', 'B', 'A', 'B'], vec!['B', 'A', 'B', 'A']],
    //         root: Node {
    //             children: HashMap::from([
    //                 (
    //                     'A',
    //                     Node {
    //                         source: 0,
    //                         start: 0,
    //                         end: 1,
    //                         children: HashMap::from([(
    //                             'B',
    //                             Node {
    //                                 source: 0,
    //                                 start: 1,
    //                                 end: 2,
    //                                 children: HashMap::from([(
    //                                     'A',
    //                                     Node {
    //                                         source: 0,
    //                                         start: 2,
    //                                         end: 3,
    //                                         children: HashMap::from([(
    //                                             'B',
    //                                             Node::terminal(0, 3, 4),
    //                                         )]),
    //                                         terminators: vec![1],
    //                                     },
    //                                 )]),
    //                                 terminators: vec![0],
    //                             },
    //                         )]),
    //                         terminators: vec![1],
    //                     },
    //                 ),
    //                 (
    //                     'B',
    //                     Node {
    //                         source: 0,
    //                         start: 1,
    //                         end: 2,
    //                         children: HashMap::from([(
    //                             'A',
    //                             Node {
    //                                 source: 0,
    //                                 start: 2,
    //                                 end: 3,
    //                                 children: HashMap::from([(
    //                                     'B',
    //                                     Node {
    //                                         source: 0,
    //                                         start: 3,
    //                                         end: 4,
    //                                         children: HashMap::from([(
    //                                             'A',
    //                                             Node::terminal(1, 3, 4),
    //                                         )]),
    //                                         terminators: vec![0],
    //                                     },
    //                                 )]),
    //                                 terminators: vec![1],
    //                             },
    //                         )]),
    //                         terminators: vec![0],
    //                     },
    //                 ),
    //             ]),
    //             ..Default::default()
    //         },
    //     };
    //     let result = SuffixTree::from(["ABAB".to_owned().chars(), "BABA".to_owned().chars()]);
    //     assert_eq!(expected, result);
    // }

    // #[test]
    // fn test_multiple_coincidence() {
    //     let expected = SuffixTree {
    //         data: vec![vec!['A', 'A', 'A'], vec!['A', 'A'], vec!['A']],
    //         root: Node {
    //             children: HashMap::from([(
    //                 'A',
    //                 Node {
    //                     source: 0,
    //                     start: 0,
    //                     end: 1,
    //                     children: HashMap::from([(
    //                         'A',
    //                         Node {
    //                             source: 0,
    //                             start: 1,
    //                             end: 2,
    //                             children: HashMap::from([('A', Node::terminal(0, 2, 3))]),
    //                             terminators: vec![0, 1],
    //                         },
    //                     )]),
    //                     terminators: vec![0, 1, 2],
    //                 },
    //             )]),
    //             ..Default::default()
    //         },
    //     };
    //     let result = SuffixTree::from([
    //         "AAA".to_owned().chars(),
    //         "AA".to_owned().chars(),
    //         "A".to_owned().chars(),
    //     ]);
    //     assert_eq!(expected, result);
    // }

    // #[test]
    // fn test_multiple_coincidence_rev() {
    //     let expected = SuffixTree {
    //         data: vec![vec!['A'], vec!['A', 'A'], vec!['A', 'A', 'A']],
    //         root: Node {
    //             children: HashMap::from([(
    //                 'A',
    //                 Node {
    //                     source: 0,
    //                     start: 0,
    //                     end: 1,
    //                     children: HashMap::from([(
    //                         'A',
    //                         Node {
    //                             source: 1,
    //                             start: 1,
    //                             end: 2,
    //                             children: HashMap::from([('A', Node::terminal(2, 2, 3))]),
    //                             terminators: vec![1, 2],
    //                         },
    //                     )]),
    //                     terminators: vec![0, 1, 2],
    //                 },
    //             )]),
    //             ..Default::default()
    //         },
    //     };
    //     let result = SuffixTree::from([
    //         "A".to_owned().chars(),
    //         "AA".to_owned().chars(),
    //         "AAA".to_owned().chars(),
    //     ]);
    //     assert_eq!(expected, result);
    // }
}
