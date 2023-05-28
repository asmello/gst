use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

#[derive(Default, Debug, PartialEq, Eq)]
struct Node<E>
where
    E: Copy + Default + Eq + Hash + Debug,
{
    index: usize,
    start: usize,
    end: usize,
    children: HashMap<E, Node<E>>,
    terminators: Vec<usize>,
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
    fn len(&self) -> usize {
        self.end - self.start
    }
    fn get<I>(&self, data: &[I], idx: usize) -> E
    where
        I: AsRef<[E]>,
    {
        data[self.index].as_ref()[self.start + idx]
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct SuffixTree<E>
where
    E: Copy + Default + Eq + Hash + Debug,
{
    data: Vec<Vec<E>>,
    root: Node<E>,
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
        Self::default()
    }

    pub fn add(&mut self, data: Vec<E>) {
        self.add_naive(data);
    }

    fn add_naive(&mut self, data: Vec<E>) {
        let t = self.data.len();
        self.data.push(data);
        let n = self.data[t].len();
        for i in 0..n {
            let mut curr = &mut self.root;
            let mut j = i;
            while j < n {
                // do we have an edge starting at data[j]?
                if let Some(child) = curr.children.get(&self.data[t][j]) {
                    for k in 1..child.len() {
                        // traverse the edge
                        if let Some(&elem) = self.data[t].get(j + k) {
                            // test if next element matches edge at same position
                            if elem != child.get(&self.data, k) {
                                // mismatch, fork the edge at k and create a new branch

                                let orig = child.get(&self.data, k);
                                let mut mid = Node::new(child.index, child.start, child.start + k);

                                let new_branch = Node::new(t, j + k, n);
                                mid.children.insert(elem, new_branch);

                                let mut child = curr.children.remove(&self.data[t][j]).unwrap();
                                child.start += k;
                                mid.children.insert(orig, child);
                                curr.children.insert(self.data[t][j], mid);

                                break; // the edge now matches to its end
                            }
                        } else {
                            // end of the data, fork the edge at k but don't create a new branch

                            let orig = child.get(&self.data, k);
                            let mut mid = Node::new(child.index, child.start, child.start + k);

                            let mut child = curr.children.remove(&self.data[t][j]).unwrap();
                            child.start += k;
                            mid.children.insert(orig, child);
                            curr.children.insert(self.data[t][j], mid);

                            break; // the edge now matches to its end
                        }
                    }
                    // now we know all match, descend into child
                    curr = curr.children.get_mut(&self.data[t][j]).unwrap();
                    j += curr.len();
                } else {
                    // no matches for prefix, insert a new edge
                    let new_node = Node::new(t, j, n);
                    curr.children.insert(self.data[t][j], new_node);
                }
            }
            curr.terminators.push(t);
        }
    }
}

impl<E> std::fmt::Display for SuffixTree<E>
where
    E: Copy + Default + Eq + Hash + Debug,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut stack: Vec<(&Node<E>, String, bool)> = Vec::with_capacity(self.root.children.len());
        for (idx, child) in self.root.children.values().enumerate() {
            stack.push((child, "".into(), idx == self.root.children.len() - 1));
            while let Some((curr, mut prefix, is_last)) = stack.pop() {
                let to_add = if is_last { "└──" } else { "├──" };
                let curr_prefix: String = prefix.chars().chain(to_add.chars()).collect();
                write!(
                    fmt,
                    "{curr_prefix}{:?}",
                    &self.data[curr.index][curr.start..curr.end],
                )?;
                if !curr.terminators.is_empty() {
                    write!(fmt, " - {:?}", curr.terminators)?;
                }
                write!(fmt, "\n")?;
                if !curr.children.is_empty() {
                    if is_last {
                        prefix.push_str("   ");
                    } else {
                        prefix.push_str("│  ");
                    }
                    for (idx, child) in curr.children.values().enumerate() {
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

    impl<E> Node<E>
    where
        E: Copy + Default + Eq + Hash + Debug,
    {
        fn terminal(index: usize, start: usize, end: usize) -> Self {
            Self {
                index,
                start,
                end,
                terminators: vec![index],
                ..Default::default()
            }
        }
    }

    #[test]
    fn test_aabccb_unique() {
        let str = "aabccb$";
        let expected = SuffixTree {
            data: vec![str.to_owned().chars().collect()],
            root: Node {
                children: HashMap::from([
                    ('$', Node::terminal(0, 6, 7)),
                    (
                        'a',
                        Node {
                            start: 0,
                            end: 1,
                            children: HashMap::from([
                                ('a', Node::terminal(0, 1, 7)), // abccb$
                                ('b', Node::terminal(0, 2, 7)), // bccb$
                            ]),
                            ..Default::default()
                        },
                    ),
                    (
                        'b',
                        Node {
                            start: 2,
                            end: 3,
                            children: HashMap::from([
                                ('$', Node::terminal(0, 6, 7)), // $
                                ('c', Node::terminal(0, 3, 7)), // ccb$
                            ]),
                            ..Default::default()
                        },
                    ),
                    (
                        'c',
                        Node {
                            start: 3,
                            end: 4,
                            children: HashMap::from([
                                ('b', Node::terminal(0, 5, 7)), // b$
                                ('c', Node::terminal(0, 4, 7)), // cb$
                            ]),
                            ..Default::default()
                        },
                    ),
                ]),
                ..Default::default()
            },
        };
        let result = SuffixTree::from([str.to_owned().chars()]);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_aabccb() {
        let str = "aabccb";
        let expected = SuffixTree {
            data: vec![str.to_owned().chars().collect()],
            root: Node {
                children: HashMap::from([
                    (
                        'a',
                        Node {
                            start: 0,
                            end: 1,
                            children: HashMap::from([
                                ('a', Node::terminal(0, 1, 6)), // abccb
                                ('b', Node::terminal(0, 2, 6)), // bccb
                            ]),
                            ..Default::default()
                        },
                    ),
                    (
                        'b',
                        Node {
                            start: 2,
                            end: 3,
                            children: HashMap::from([
                                ('c', Node::terminal(0, 3, 6)), // ccb
                            ]),
                            terminators: vec![0],
                            ..Default::default()
                        },
                    ),
                    (
                        'c',
                        Node {
                            start: 3,
                            end: 4,
                            children: HashMap::from([
                                ('b', Node::terminal(0, 5, 6)), // b
                                ('c', Node::terminal(0, 4, 6)), // cb
                            ]),
                            ..Default::default()
                        },
                    ),
                ]),
                ..Default::default()
            },
        };
        let result = SuffixTree::from([str.to_owned().chars()]);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_multiple() {
        let expected = SuffixTree {
            data: vec![vec!['A', 'B', 'A', 'B'], vec!['B', 'A', 'B', 'A']],
            root: Node {
                children: HashMap::from([
                    (
                        'A',
                        Node {
                            index: 0,
                            start: 0,
                            end: 1,
                            children: HashMap::from([(
                                'B',
                                Node {
                                    index: 0,
                                    start: 1,
                                    end: 2,
                                    children: HashMap::from([(
                                        'A',
                                        Node {
                                            index: 0,
                                            start: 2,
                                            end: 3,
                                            children: HashMap::from([(
                                                'B',
                                                Node::terminal(0, 3, 4),
                                            )]),
                                            terminators: vec![1],
                                        },
                                    )]),
                                    terminators: vec![0],
                                },
                            )]),
                            terminators: vec![1],
                        },
                    ),
                    (
                        'B',
                        Node {
                            index: 0,
                            start: 1,
                            end: 2,
                            children: HashMap::from([(
                                'A',
                                Node {
                                    index: 0,
                                    start: 2,
                                    end: 3,
                                    children: HashMap::from([(
                                        'B',
                                        Node {
                                            index: 0,
                                            start: 3,
                                            end: 4,
                                            children: HashMap::from([(
                                                'A',
                                                Node::terminal(1, 3, 4),
                                            )]),
                                            terminators: vec![0],
                                        },
                                    )]),
                                    terminators: vec![1],
                                },
                            )]),
                            terminators: vec![0],
                        },
                    ),
                ]),
                ..Default::default()
            },
        };
        let result = SuffixTree::from(["ABAB".to_owned().chars(), "BABA".to_owned().chars()]);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_multiple_coincidence() {
        let expected = SuffixTree {
            data: vec![vec!['A', 'A', 'A'], vec!['A', 'A'], vec!['A']],
            root: Node {
                children: HashMap::from([(
                    'A',
                    Node {
                        index: 0,
                        start: 0,
                        end: 1,
                        children: HashMap::from([(
                            'A',
                            Node {
                                index: 0,
                                start: 1,
                                end: 2,
                                children: HashMap::from([('A', Node::terminal(0, 2, 3))]),
                                terminators: vec![0, 1],
                            },
                        )]),
                        terminators: vec![0, 1, 2],
                    },
                )]),
                ..Default::default()
            },
        };
        let result = SuffixTree::from([
            "AAA".to_owned().chars(),
            "AA".to_owned().chars(),
            "A".to_owned().chars(),
        ]);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_multiple_coincidence_rev() {
        let expected = SuffixTree {
            data: vec![vec!['A'], vec!['A', 'A'], vec!['A', 'A', 'A']],
            root: Node {
                children: HashMap::from([(
                    'A',
                    Node {
                        index: 0,
                        start: 0,
                        end: 1,
                        children: HashMap::from([(
                            'A',
                            Node {
                                index: 1,
                                start: 1,
                                end: 2,
                                children: HashMap::from([('A', Node::terminal(2, 2, 3))]),
                                terminators: vec![1, 2],
                            },
                        )]),
                        terminators: vec![0, 1, 2],
                    },
                )]),
                ..Default::default()
            },
        };
        let result = SuffixTree::from([
            "A".to_owned().chars(),
            "AA".to_owned().chars(),
            "AAA".to_owned().chars(),
        ]);
        assert_eq!(expected, result);
    }
}
