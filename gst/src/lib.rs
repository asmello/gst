use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

const NULL: usize = 0;
const ROOT: usize = 1;

#[derive(Default, Debug, PartialEq, Eq)]
struct Node<E>
where
    E: Copy + Default + Eq + Hash + Debug,
{
    index: usize,
    start: usize,
    end: usize,
    edges: HashMap<E, usize>,
}

impl<E> Node<E>
where
    E: Copy + Default + Eq + Hash + Debug,
{
    fn new(index: usize, start: usize, end: usize) -> Self {
        Self {
            index,
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
    nodes: Vec<Node<E>>,
    links: HashMap<usize, usize>,
    terminators: HashMap<usize, usize>,
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

    // fn get_from_tree(&self, node: usize, edge: E, offset: usize) -> Option<E> {
    //     let node = &self.nodes[node];
    //     node.edges
    //         .get(&edge)
    //         .map(|&child| self.data[node.index][self.nodes[child].start + offset])
    // }
    // fn get_child(&self, node: usize, edge: E) -> Option<usize> {
    //     self.nodes[node].edges.get(&edge).copied()
    // }

    fn new_node(&mut self, index: usize, start: usize, end: usize) -> usize {
        self.nodes.push(Node::new(index, start, end));
        self.nodes.len()
    }

    fn add_edge(&mut self, src: usize, key: E, dst: usize) {
        self.nodes[src - 1].edges.insert(key, dst);
    }

    fn edge_exists(&self, src: usize, key: E) -> bool {
        match src {
            0 => true,
            n => self.nodes[n - 1].edges.contains_key(&key),
        }
    }

    fn edge(&self, src: usize, key: E) -> usize {
        match src {
            0 => 1,
            n => self.nodes[n - 1].edges[&key],
        }
    }

    fn link(&self, node: usize) -> usize {
        self.links[&node]
    }

    fn add_link(&mut self, src: usize, dst: usize) {
        self.links.insert(src, dst);
    }
    // r can refer to any implicit or explicit state
    // reference pair (s, w) = r has s as an explicit state and w a string starting at s
    // (s, w) is canonical if s is the closest ancestor to r (when r is explicit, w is empty)
    // (s, w) can also be represented (s, (k, p)) for two points (k, p) into the string

    fn add_ukkonen(&mut self, data: Vec<E>) {
        let s = self.data.len();
        let n = data.len();
        self.data.push(data);

        let mut top = ROOT;
        let mut prev = NULL;
        let mut curr;

        for i in 0..n {
            let t_i = self.data[s][i];
            curr = top;
            while !self.edge_exists(curr, t_i) {
                let new = self.new_node(s, i, i + 1);
                self.add_edge(curr, t_i, new);
                if curr != top {
                    self.add_link(prev, new);
                }
                prev = new;
                curr = self.link(curr);
            }
            self.add_link(prev, self.edge(curr, t_i));
            top = self.edge(top, t_i);
        }
    }

    // fn add_ukkonen(&mut self, data: Vec<E>) {
    //     let i = self.data.len();
    //     let n = data.len();
    //     self.data.push(data);
    //     let mut p = 0;
    //     let mut active_node = 0;
    //     let mut active_edge = None;
    //     let mut active_length = 0;
    //     let mut remainder = 1;
    //     let mut previous_insertion = None;
    //     while p < n {
    //         let curr_data = self.data[i][p];
    //         if active_edge.is_none() && self.edge_exists(active_node, curr_data) {
    //             active_edge = Some(curr_data);
    //             active_length += 1;
    //             remainder += 1;
    //             p += 1;
    //         } else if let Some(edge) = active_edge {
    //             if let Some(curr_tree) = self.get_from_tree(active_node, edge, active_length) {
    //                 if curr_tree == curr_data {
    //                     active_length += 1;
    //                     remainder += 1;
    //                     p += 1;
    //                 } else {
    //                     let child = self.get_child(active_node, edge).unwrap();
    //                     let mid = self.nodes.len();
    //                     self.nodes.push(Node::new(
    //                         i,
    //                         self.nodes[child].start,
    //                         self.nodes[child].start + active_length,
    //                     ));
    //                     let new = self.nodes.len();
    //                     self.nodes.push(Node::new(i, p, n));
    //                     self.nodes[mid].edges.insert(curr_data, new);
    //                     self.nodes[mid].edges.insert(curr_tree, child);
    //                     self.nodes[child].start += active_length;
    //                     self.nodes[active_node].edges.insert(edge, mid);
    //                     remainder -= 1;
    //                     // rule 1
    //                     if active_node == 0 {
    //                         active_length -= 1;
    //                         active_edge = Some(self.data[i][p + 1 - remainder]);
    //                     }
    //                     if let Some(previous) = previous_insertion {
    //                         self.links.insert(previous, mid);
    //                     }
    //                     previous_insertion = Some(mid);
    //                 }
    //             } else {
    //                 let new = self.nodes.len();
    //                 self.nodes.push(Node::new(i, p, n));
    //                 self.nodes[active_node].edges.insert(curr_data, new);
    //                 active_edge = None;
    //                 remainder -= 1;
    //                 p += 1;
    //             }
    //         } else {
    //             let new = self.nodes.len();
    //             self.nodes.push(Node::new(i, p, n));
    //             self.nodes[active_node].edges.insert(curr_data, new);
    //             p += 1;
    //         }
    //     }
    //     // let mut active_node = 0;
    //     // // for each prefix S[0..i] = P
    //     // for i in 0..n {
    //     //     // for each suffix S[j..i] of P
    //     //     for j in 0..=i {
    //     //         let mut curr = active_node; // find path S[j..i] in tree
    //     //         let curr_data = self.data[0][j];
    //     //         if j == 0 {
    //     //             if let Some(&child) = self.nodes[curr].edges.get(&curr_data) {
    //     //                 self.nodes[child].end += 1;
    //     //             } else {
    //     //                 let new = self.nodes.len();
    //     //                 self.nodes.push(Node::new(0, 0, i + 1));
    //     //                 self.nodes[curr].edges.insert(curr_data, new);
    //     //                 active_node = new;
    //     //             }
    //     //         } else if j == 1 {
    //     //             if active_node == 0 {
    //     //             }
    //     //         }
    //     //         let mut k = j;
    //     //         'outer: while k <= i {
    //     //             if let Some(&child) = self.nodes[curr].edges.get(&self.data[0][k]) {
    //     //                 let child_node = &self.nodes[child];
    //     //                 // we have an edge starting with S[k], traverse it
    //     //                 // edge case: child.end == n used for leaf nodes, but we should compute length from current end
    //     //                 // let m = (child_node.end - child_node.start).min(i + 1 - child_node.start);
    //     //                 let m = child_node.end - child_node.start;
    //     //                 for u in 1..m {
    //     //                     if k + u > i {
    //     //                         // reached end of suffix S[j..=i], apply rule 3 - do nothing
    //     //                         break 'outer; // extension finished
    //     //                     }
    //     //                     let child_value = self.data[child_node.index][child_node.start + u];
    //     //                     if self.data[0][k + u] != child_value {
    //     //                         // S[k + u] is not in path, apply rule 2b - create new branch
    //     //                         let mid = self.nodes.len();
    //     //                         self.nodes.push(Node::new(
    //     //                             child_node.index,
    //     //                             child_node.start,
    //     //                             child_node.start + u,
    //     //                         ));
    //     //                         let new = self.nodes.len();
    //     //                         self.nodes.push(Node::new(0, k + u, i + 1));
    //     //                         self.nodes[mid].edges.insert(self.data[0][k + u], new);
    //     //                         self.nodes[child].start += u;
    //     //                         self.nodes[mid].edges.insert(child_value, child);
    //     //                         self.nodes[curr].edges.insert(self.data[0][k], mid);
    //     //                         break 'outer; // extension finished
    //     //                     }
    //     //                 }
    //     //                 // reached the end of edge, all elements match
    //     //                 if child_node.edges.is_empty() {
    //     //                     // apply extension rule 1 - extend edge with S[i] - done implicitly
    //     //                     self.nodes[child].end += 1;
    //     //                     break; // extension finished
    //     //                 } else {
    //     //                     // continue down the child at edge's end
    //     //                     curr = *self.nodes[curr].edges.get(&self.data[0][k]).unwrap();
    //     //                     k += m;
    //     //                 }
    //     //             } else {
    //     //                 // extension rule 2a - create new edge for S[k..=i]
    //     //                 let new = self.nodes.len();
    //     //                 self.nodes.push(Node::new(0, k, i + 1));
    //     //                 self.nodes[curr].edges.insert(self.data[0][k], new);
    //     //                 break; // extension finished
    //     //             }
    //     //         }
    //     //         // If we get here, this is also rule 3 - do nothing.
    //     //         // This can happen if the child we were exploring at k == i was an internal node of length exactly 1.
    //     //         // In that case we don't enter the u-loop, so we don't reach the normal exit point for rule 3. Instead,
    //     //         // we descend into the child, but since we always advance k by the child's length, the while loop stops
    //     //         // naturally.
    //     //     }
    // }
}

impl<E> std::fmt::Display for SuffixTree<E>
where
    E: Copy + Default + Eq + Hash + Debug,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "1\n")?;
        let n = self.nodes[0].edges.len();
        let mut stack: Vec<(usize, String, bool)> = Vec::with_capacity(n);
        for (idx, &child) in self.nodes[0].edges.values().enumerate() {
            stack.push((child, "".into(), idx == n - 1));
            while let Some((curr, mut prefix, is_last)) = stack.pop() {
                let to_add = if is_last { "└──" } else { "├──" };
                let curr_prefix: String = prefix.chars().chain(to_add.chars()).collect();
                let curr_node = &self.nodes[curr - 1];
                write!(
                    fmt,
                    "{curr_prefix}{curr}:{:?}",
                    &self.data[curr_node.index][curr_node.start..curr_node.end],
                )?;
                if self.terminators.contains_key(&curr) {
                    write!(fmt, "~{:?}", self.terminators[&curr])?;
                }
                if self.links.contains_key(&curr) {
                    write!(fmt, " ➔ {}", self.links[&curr])?;
                }
                write!(fmt, "\n")?;
                if !self.nodes[curr - 1].edges.is_empty() {
                    if is_last {
                        prefix.push_str("   ");
                    } else {
                        prefix.push_str("│  ");
                    }
                    for (idx, &child) in self.nodes[curr - 1].edges.values().enumerate() {
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

    #[test]
    fn test() {
        let str = "cacao";
        let result = SuffixTree::from([str.to_owned().chars()]);
        println!("{result}");
    }

    // #[test]
    // fn test_aabccb_unique() {
    //     let str = "aabccb$";
    //     let expected = SuffixTree {
    //         data: vec![str.to_owned().chars().collect()],
    //         root: Node {
    //             children: HashMap::from([
    //                 ('$', Node::terminal(0, 6, 7)),
    //                 (
    //                     'a',
    //                     Node {
    //                         start: 0,
    //                         end: 1,
    //                         children: HashMap::from([
    //                             ('a', Node::terminal(0, 1, 7)), // abccb$
    //                             ('b', Node::terminal(0, 2, 7)), // bccb$
    //                         ]),
    //                         ..Default::default()
    //                     },
    //                 ),
    //                 (
    //                     'b',
    //                     Node {
    //                         start: 2,
    //                         end: 3,
    //                         children: HashMap::from([
    //                             ('$', Node::terminal(0, 6, 7)), // $
    //                             ('c', Node::terminal(0, 3, 7)), // ccb$
    //                         ]),
    //                         ..Default::default()
    //                     },
    //                 ),
    //                 (
    //                     'c',
    //                     Node {
    //                         start: 3,
    //                         end: 4,
    //                         children: HashMap::from([
    //                             ('b', Node::terminal(0, 5, 7)), // b$
    //                             ('c', Node::terminal(0, 4, 7)), // cb$
    //                         ]),
    //                         ..Default::default()
    //                     },
    //                 ),
    //             ]),
    //             ..Default::default()
    //         },
    //     };
    //     let result = SuffixTree::from([str.to_owned().chars()]);
    //     assert_eq!(expected, result);
    // }

    // #[test]
    // fn test_aabccb() {
    //     let str = "aabccb";
    //     let expected = SuffixTree {
    //         data: vec![str.to_owned().chars().collect()],
    //         root: Node {
    //             children: HashMap::from([
    //                 (
    //                     'a',
    //                     Node {
    //                         start: 0,
    //                         end: 1,
    //                         children: HashMap::from([
    //                             ('a', Node::terminal(0, 1, 6)), // abccb
    //                             ('b', Node::terminal(0, 2, 6)), // bccb
    //                         ]),
    //                         ..Default::default()
    //                     },
    //                 ),
    //                 (
    //                     'b',
    //                     Node {
    //                         start: 2,
    //                         end: 3,
    //                         children: HashMap::from([
    //                             ('c', Node::terminal(0, 3, 6)), // ccb
    //                         ]),
    //                         terminators: vec![0],
    //                         ..Default::default()
    //                     },
    //                 ),
    //                 (
    //                     'c',
    //                     Node {
    //                         start: 3,
    //                         end: 4,
    //                         children: HashMap::from([
    //                             ('b', Node::terminal(0, 5, 6)), // b
    //                             ('c', Node::terminal(0, 4, 6)), // cb
    //                         ]),
    //                         ..Default::default()
    //                     },
    //                 ),
    //             ]),
    //             ..Default::default()
    //         },
    //     };
    //     let result = SuffixTree::from([str.to_owned().chars()]);
    //     assert_eq!(expected, result);
    // }

    // #[test]
    // fn test_multiple() {
    //     let expected = SuffixTree {
    //         data: vec![vec!['A', 'B', 'A', 'B'], vec!['B', 'A', 'B', 'A']],
    //         root: Node {
    //             children: HashMap::from([
    //                 (
    //                     'A',
    //                     Node {
    //                         index: 0,
    //                         start: 0,
    //                         end: 1,
    //                         children: HashMap::from([(
    //                             'B',
    //                             Node {
    //                                 index: 0,
    //                                 start: 1,
    //                                 end: 2,
    //                                 children: HashMap::from([(
    //                                     'A',
    //                                     Node {
    //                                         index: 0,
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
    //                         index: 0,
    //                         start: 1,
    //                         end: 2,
    //                         children: HashMap::from([(
    //                             'A',
    //                             Node {
    //                                 index: 0,
    //                                 start: 2,
    //                                 end: 3,
    //                                 children: HashMap::from([(
    //                                     'B',
    //                                     Node {
    //                                         index: 0,
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
    //                     index: 0,
    //                     start: 0,
    //                     end: 1,
    //                     children: HashMap::from([(
    //                         'A',
    //                         Node {
    //                             index: 0,
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
    //                     index: 0,
    //                     start: 0,
    //                     end: 1,
    //                     children: HashMap::from([(
    //                         'A',
    //                         Node {
    //                             index: 1,
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
