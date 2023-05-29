use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

const ROOT: usize = 1;

#[derive(Debug, PartialEq, Eq)]
struct Node {
    source: usize,
    start: usize,
    end: usize,
}

impl Node {
    fn new(source: usize, start: usize, end: usize) -> Self {
        Self { source, start, end }
    }

    fn len(&self) -> usize {
        if self.end == usize::MAX {
            usize::MAX
        } else {
            self.end - self.start + 1
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum Token<E>
where
    E: Copy + Eq + Hash + Debug,
{
    Element(E),
    Terminator(usize),
}

impl<E> Token<E>
where
    E: Copy + Eq + Hash + Debug,
{
    fn is_elem(&self) -> bool {
        if let Self::Element(_) = self {
            true
        } else {
            false
        }
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct GeneralizedSuffixTree<E>
where
    E: Copy + Eq + Hash + Debug,
{
    elems: Vec<Vec<E>>,
    nodes: Vec<Node>,
    edges: HashMap<usize, HashMap<Token<E>, usize>>,
    links: HashMap<usize, usize>,
}

impl<E, I, const N: usize> From<[I; N]> for GeneralizedSuffixTree<E>
where
    I: IntoIterator<Item = E>,
    E: Copy + Default + Eq + Hash + Debug,
{
    fn from(arr: [I; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<E, I> FromIterator<I> for GeneralizedSuffixTree<E>
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
            st.insert(sequence.into_iter().collect());
        }
        st
    }
}

impl<E> GeneralizedSuffixTree<E>
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

    pub fn insert(&mut self, data: Vec<E>) {
        self.run_ukkonen(data);
    }

    fn new_node(&mut self, source: usize, start: usize, end: usize) -> usize {
        self.nodes.push(Node::new(source, start, end));
        self.nodes.len()
    }

    fn add_edge(&mut self, src: usize, key: Token<E>, dst: usize) {
        if src != ROOT || key.is_elem() {
            // no need to add terminators to root node (empty strings should be always matched implicitly!)
            self.edges.entry(src).or_default().insert(key, dst);
        }
    }

    fn edge_exists(&self, src: usize, key: Token<E>) -> bool {
        match src {
            0 => true,
            n => self
                .edges
                .get(&n)
                .map(|edges| edges.contains_key(&key))
                .unwrap_or_default(),
        }
    }

    fn edge(&self, node: usize, key: Token<E>) -> usize {
        match node {
            0 => 1,
            n => self.edges[&n][&key],
        }
    }

    fn get_edge(&self, node: usize, elem: E) -> Option<usize> {
        match node {
            0 => Some(1),
            n => self
                .edges
                .get(&n)
                .map(|edges| edges.get(&Token::Element(elem)))
                .flatten()
                .copied(),
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

    fn get(&self, source: usize, pos: usize) -> Token<E> {
        if pos <= self.elems[source].len() {
            Token::Element(self.elems[source][pos - 1])
        } else {
            Token::Terminator(source)
        }
    }

    fn get_elem(&self, source: usize, pos: usize) -> E {
        self.elems[source][pos - 1] // panics if index is outside of range
    }

    fn end(&self, source: usize) -> usize {
        self.elems[source].len()
    }

    fn add_link(&mut self, src: usize, dst: usize) {
        self.links.insert(src, dst);
    }

    fn run_ukkonen(&mut self, data: Vec<E>) {
        let source = self.elems.len();
        self.elems.push(data);

        let mut node = ROOT;
        let mut start = 1;
        for i in 1..=self.end(source) + 1 {
            (node, start) = self.update(source, node, start, i);
            (node, start) = self.canonize(source, node, start, i);
        }
    }

    fn update(
        &mut self,
        source: usize,
        mut node: usize,
        mut start: usize,
        i: usize,
    ) -> (usize, usize) {
        let t_i = self.get(source, i);
        let mut prev = ROOT;
        let (mut endpoint, mut curr) = self.test_and_split(source, node, start, i - 1, t_i);
        while !endpoint {
            let new = self.new_node(source, i, usize::MAX);
            self.add_edge(curr, t_i, new);
            if prev != ROOT {
                self.add_link(prev, curr);
            }
            prev = curr;
            (node, start) = self.canonize(source, self.link(node), start, i - 1);
            (endpoint, curr) = self.test_and_split(source, node, start, i - 1, t_i);
        }
        if prev != ROOT {
            self.add_link(prev, node);
        }
        (node, start)
    }

    fn test_and_split(
        &mut self,
        source: usize,
        node: usize,
        start: usize,
        end: usize,
        t: Token<E>,
    ) -> (bool, usize) {
        if start <= end {
            let t_start = self.get(source, start);
            let child = self.edge(node, t_start);
            let t_mid = self.get(
                self.node(child).source,
                1 + self.node(child).start + end - start,
            );
            if t == t_mid {
                (true, node)
            } else {
                let mid = self.new_node(
                    self.node(child).source,
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

    fn canonize(
        &mut self,
        source: usize,
        mut node: usize,
        mut start: usize,
        end: usize,
    ) -> (usize, usize) {
        if end >= start {
            let mut t_start = self.get(source, start);
            let mut child = self.edge(node, t_start);
            while end >= start && self.node(child).end - self.node(child).start <= end - start {
                start += self.node(child).end - self.node(child).start + 1;
                node = child;
                if start <= end {
                    t_start = self.get(source, start);
                    child = self.edge(node, t_start);
                }
            }
        }
        (node, start)
    }

    pub fn find_all(&self, pattern: &[E]) -> Vec<(usize, usize)> {
        let n = pattern.len();
        let mut matches = Vec::new();
        let mut stack = Vec::from([(ROOT, 0)]);
        'outer: while let Some((node_id, i)) = stack.pop() {
            if let Some(child_id) = self.get_edge(node_id, pattern[i]) {
                let child = self.node(child_id);
                let m: usize = child.len().min(self.end(child.source) - child.start + 1);
                for j in 1..m {
                    if i + j == n {
                        // found a match
                        matches.extend(self.find_positions(child_id, i));
                        continue 'outer;
                    }

                    if self.get_elem(child.source, child.start + j) != pattern[i + j] {
                        continue 'outer;
                    }
                }
                if i + m == n {
                    // found a match
                    matches.extend(self.find_positions(child_id, i));
                } else {
                    stack.push((child_id, i + m));
                }
            }
        }
        matches
    }

    fn find_positions(&self, node: usize, depth: usize) -> Vec<(usize, usize)> {
        let mut pos = Vec::new();
        let mut stack = Vec::from([(node, depth)]);
        while let Some((node_id, depth)) = stack.pop() {
            let node = self.node(node_id);
            let m = node.len().min(self.end(node.source) - node.start + 1);
            if let Some(edges) = self.edges.get(&node_id) {
                for (&token, &child) in edges {
                    match token {
                        Token::Element(_) => {
                            stack.push((child, depth + m));
                        }
                        Token::Terminator(source) => {
                            pos.push((source, self.end(source) - depth - m));
                        }
                    }
                }
            } else {
                pos.push((node.source, self.end(node.source) - depth - m));
            }
        }
        pos
    }
}

impl<E> std::fmt::Display for GeneralizedSuffixTree<E>
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
                let span = if self.node(curr).start > self.end(self.node(curr).source) {
                    format!("T")
                } else {
                    format!(
                        "{:?}",
                        &self.elems[self.node(curr).source][self.node(curr).start - 1
                            ..self.node(curr).end.min(self.end(self.node(curr).source))]
                    )
                };
                write!(fmt, "{curr_prefix} {curr} {span}",)?;
                if !self.edges.contains_key(&curr) {
                    write!(fmt, ":{}", self.node(curr).source)?;
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

    #[test]
    fn test_find_abab() {
        let result = GeneralizedSuffixTree::from(["abab".to_owned().chars()]);
        let mut pos = result.find_all(&['a']);
        pos.sort();
        assert_eq!(pos, [(0, 0), (0, 2)]);
        pos = result.find_all(&['b', 'a']);
        pos.sort();
        assert_eq!(pos, [(0, 1)]);
        pos = result.find_all(&['b']);
        pos.sort();
        assert_eq!(pos, [(0, 1), (0, 3)]);
        pos = result.find_all(&['c']);
        assert_eq!(pos, []);
        pos = result.find_all(&['b', 'c']);
        assert_eq!(pos, []);
        pos = result.find_all(&['b', 'a', 'n', 'a', 'n', 'a']);
        assert_eq!(pos, []);
    }

    #[test]
    fn test_find_banana() {
        let result = GeneralizedSuffixTree::from(["banana".to_owned().chars()]);
        let mut pos = result.find_all(&['b', 'a', 'n', 'a', 'n', 'a']);
        pos.sort();
        assert_eq!(pos, [(0, 0)]);
        pos = result.find_all(&['n', 'a']);
        pos.sort();
        assert_eq!(pos, [(0, 2), (0, 4)]);
        pos = result.find_all(&['a']);
        pos.sort();
        assert_eq!(pos, [(0, 1), (0, 3), (0, 5)]);
        pos = result.find_all(&['a', 'n', 'a']);
        pos.sort();
        assert_eq!(pos, [(0, 1), (0, 3)]);
    }

    #[test]
    fn test_find_multiple_sources() {
        let result =
            GeneralizedSuffixTree::from(["banana".to_owned().chars(), "anna".to_owned().chars()]);
        let mut pos = result.find_all(&['b', 'a', 'n', 'a', 'n', 'a']);
        pos.sort();
        assert_eq!(pos, [(0, 0)]);
        pos = result.find_all(&['a', 'n', 'n', 'a']);
        pos.sort();
        assert_eq!(pos, [(1, 0)]);
        pos = result.find_all(&['n', 'a']);
        pos.sort();
        assert_eq!(pos, [(0, 2), (0, 4), (1, 2)]);
        pos = result.find_all(&['a']);
        pos.sort();
        assert_eq!(pos, [(0, 1), (0, 3), (0, 5), (1, 0), (1, 3)]);
    }

    #[test]
    fn test_find_positions() {
        let result = GeneralizedSuffixTree::from(["abab".to_owned().chars()]);
        let mut pos = result.find_positions(4, 0);
        pos.sort();
        assert_eq!(pos, [(0, 0), (0, 2)]);
        pos = result.find_positions(3, 1);
        pos.sort();
        assert_eq!(pos, [(0, 1)]);
        pos = result.find_positions(6, 0);
        pos.sort();
        assert_eq!(pos, [(0, 1), (0, 3)]);
    }

    // TODO: load expected tree from string representation

    // #[test]
    // fn test_aabccb_unique() {
    //     let str = "aabccb$";
    //     let result = SuffixTree::from([str.to_owned().chars()]);
    //     println!("{result}");
    // }

    // #[test]
    // fn test_aabccb() {
    //     let str = "aabccb";
    //     let result = SuffixTree::from([str.to_owned().chars()]);
    //     println!("{result}");
    // }

    // #[test]
    // fn test_multiple() {
    //     let result = SuffixTree::from(["ABAB".to_owned().chars(), "BABA".to_owned().chars()]);
    //     println!("{result}");
    // }

    // #[test]
    // fn test_multiple_coincidence() {
    //     let result = SuffixTree::from([
    //         "AAA".to_owned().chars(),
    //         "AA".to_owned().chars(),
    //         "A".to_owned().chars(),
    //     ]);
    //     println!("{result}");
    // }

    // #[test]
    // fn test_multiple_coincidence_rev() {
    //     let result = SuffixTree::from([
    //         "A".to_owned().chars(),
    //         "AA".to_owned().chars(),
    //         "AAA".to_owned().chars(),
    //     ]);
    //     println!("{result}");
    // }
}
