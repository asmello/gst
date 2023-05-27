use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash};

#[derive(Default, Debug, Eq)]
struct Node<E, S = RandomState>
where
    E: Copy + Default + Eq + Hash + Debug,
    S: BuildHasher,
{
    start: usize,
    end: usize,
    children: HashMap<E, Node<E, S>, S>,
    terminators: Vec<usize>,
}

impl<E, S> PartialEq for Node<E, S>
where
    E: Copy + Default + Eq + Hash + Debug,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start
            && self.end == other.end
            && self.children == other.children
            && self.terminators == other.terminators
    }
}

impl<E, S> Node<E, S>
where
    E: Copy + Default + Eq + Hash + Debug,
    S: BuildHasher,
{
    fn with_hasher(start: usize, end: usize, hasher: S) -> Self {
        Self {
            start,
            end,
            children: HashMap::with_hasher(hasher),
            terminators: Vec::new(),
        }
    }
    fn default_with_hasher(hasher: S) -> Self {
        Self {
            start: 0,
            end: 0,
            children: HashMap::with_hasher(hasher),
            terminators: Vec::new(),
        }
    }
    fn terminal_with_hasher(start: usize, end: usize, terminator: usize, hasher: S) -> Self {
        Self {
            start,
            end,
            children: HashMap::with_hasher(hasher),
            terminators: vec![terminator],
        }
    }
    fn len(&self) -> usize {
        self.end - self.start
    }
    fn get(&self, data: &[E], idx: usize) -> E {
        data[self.start + idx]
    }
}

#[derive(Debug, Eq)]
pub struct SuffixTree<E, S = RandomState>
where
    E: Copy + Default + Eq + Hash + Debug,
    S: BuildHasher,
{
    data: Vec<E>,
    root: Node<E, S>,
}

impl<E, S> PartialEq for SuffixTree<E, S>
where
    E: Copy + Default + Eq + Hash + Debug,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.root == other.root
    }
}

impl<E, S> SuffixTree<E, S>
where
    E: Copy + Default + Eq + Hash + Debug,
    S: BuildHasher + Clone,
{
    pub fn with_hasher(data: Vec<E>, hasher: S) -> Self {
        Self::build_naive(data, hasher)
    }

    fn build_naive(data: Vec<E>, hasher: S) -> Self {
        let mut root = Node::default_with_hasher(hasher.clone());
        if data.is_empty() {
            return Self { data, root };
        }
        let n = data.len();
        for i in 0..n {
            let mut curr = &mut root;
            let mut j = i;
            while j < n {
                // do we have an edge starting at data[j]?
                if let Some(child) = curr.children.get(&data[j]) {
                    for k in 1..child.len() {
                        // traverse the edge
                        if let Some(&elem) = data.get(j + k) {
                            // test if next element matches edge at same position
                            if elem != child.get(&data, k) {
                                // mismatch, fork the edge at k and create a new branch

                                let orig = child.get(&data, k);
                                let mut mid =
                                    Node::with_hasher(child.start, child.start + k, hasher.clone());

                                let new_branch =
                                    Node::terminal_with_hasher(j + k, n, 0, hasher.clone());
                                mid.children.insert(elem, new_branch);

                                let mut child = curr.children.remove(&data[j]).unwrap();
                                child.start += k;
                                mid.children.insert(orig, child);
                                curr.children.insert(data[j], mid);

                                break; // the edge now matches to its end
                            }
                        } else {
                            // end of the data, fork the edge at k but don't create a new branch

                            let orig = child.get(&data, k);
                            let mut mid = Node::terminal_with_hasher(
                                child.start,
                                child.start + k,
                                0,
                                hasher.clone(),
                            );

                            let mut child = curr.children.remove(&data[j]).unwrap();
                            child.start += k;
                            mid.children.insert(orig, child);
                            curr.children.insert(data[j], mid);

                            break; // the edge now matches to its end
                        }
                    }
                    // now we know all match, descend into child
                    curr = curr.children.get_mut(&data[j]).unwrap();
                    j += curr.len();
                } else {
                    // no matches for prefix, insert a new edge
                    let new_node = Node::terminal_with_hasher(j, n, 0, hasher.clone());
                    curr.children.insert(data[j], new_node);
                }
            }
        }
        Self { data, root }
    }
}

impl<E> SuffixTree<E, RandomState>
where
    E: Copy + Default + Eq + Hash + Debug,
{
    pub fn new(data: Vec<E>) -> Self {
        Self::build_naive(data, RandomState::new())
    }
}

impl<E, S> std::fmt::Display for SuffixTree<E, S>
where
    E: Copy + Default + Eq + Hash + Debug,
    S: BuildHasher,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut stack: Vec<(&Node<E, S>, String, bool)> =
            Vec::with_capacity(self.root.children.len());
        for (idx, child) in self.root.children.values().enumerate() {
            stack.push((child, "".into(), idx == self.root.children.len() - 1));
            while let Some((curr, mut prefix, is_last)) = stack.pop() {
                let to_add = if is_last { "└──" } else { "├──" };
                let curr_prefix: String = prefix.chars().chain(to_add.chars()).collect();
                write!(fmt, "{curr_prefix}{:?}", &self.data[curr.start..curr.end],)?;
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
    use std::collections::hash_map::DefaultHasher;

    use super::*;

    impl<E> Node<E, RandomState>
    where
        E: Copy + Default + Eq + Hash + Debug,
    {
        fn terminal(start: usize, end: usize, terminator: usize) -> Self {
            Self {
                start,
                end,
                terminators: vec![terminator],
                ..Default::default()
            }
        }
    }

    #[derive(Clone)]
    struct DeterministicHasher;

    impl BuildHasher for DeterministicHasher {
        type Hasher = DefaultHasher;

        fn build_hasher(&self) -> Self::Hasher {
            DefaultHasher::default()
        }
    }

    #[test]
    fn test_aabccb_unique() {
        let str = "aabccb$";
        let expected = SuffixTree {
            data: str.to_owned().chars().collect(),
            root: Node {
                children: HashMap::from([
                    ('$', Node::terminal(6, 7, 0)),
                    (
                        'a',
                        Node {
                            start: 0,
                            end: 1,
                            children: HashMap::from([
                                ('a', Node::terminal(1, 7, 0)), // abccb$
                                ('b', Node::terminal(2, 7, 0)), // bccb$
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
                                ('$', Node::terminal(6, 7, 0)), // $
                                ('c', Node::terminal(3, 7, 0)), // ccb$
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
                                ('b', Node::terminal(5, 7, 0)), // b$
                                ('c', Node::terminal(4, 7, 0)), // cb$
                            ]),
                            ..Default::default()
                        },
                    ),
                ]),
                ..Default::default()
            },
        };
        let result = SuffixTree::new(str.to_owned().chars().collect());
        assert_eq!(expected, result);
    }

    #[test]
    fn test_display() {
        let expected = include_str!("../resources/out.txt").to_owned();
        let result = format!(
            "{}",
            SuffixTree::with_hasher("aabccb$".to_owned().chars().collect(), DeterministicHasher)
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_aabccb() {
        let str = "aabccb";
        let expected = SuffixTree {
            data: str.to_owned().chars().collect(),
            root: Node {
                children: HashMap::from([
                    (
                        'a',
                        Node {
                            start: 0,
                            end: 1,
                            children: HashMap::from([
                                ('a', Node::terminal(1, 6, 0)), // abccb
                                ('b', Node::terminal(2, 6, 0)), // bccb
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
                                ('c', Node::terminal(3, 6, 0)), // ccb
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
                                ('b', Node::terminal(5, 6, 0)), // b
                                ('c', Node::terminal(4, 6, 0)), // cb
                            ]),
                            ..Default::default()
                        },
                    ),
                ]),
                ..Default::default()
            },
        };
        let result = SuffixTree::new(str.to_owned().chars().collect());
        assert_eq!(expected, result);
    }
}
