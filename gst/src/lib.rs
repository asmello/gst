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

pub struct NodeView<'a, E>
where
    E: Copy + Eq + Hash + Debug,
{
    pub span: &'a [E],
    pub edges: HashMap<E, usize>,
    pub terminators: Vec<usize>,
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
        let it = iter.into_iter();
        let mut st = Self::with_capacity(it.size_hint().0);
        for sequence in it {
            st.insert(sequence);
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

    fn with_capacity(capacity: usize) -> Self {
        Self {
            elems: Vec::with_capacity(capacity),
            nodes: vec![Node::new(0, 0, 0)],
            links: HashMap::from([(1, 0)]),
            ..Default::default()
        }
    }

    pub fn get_node(&self, id: usize) -> Option<NodeView<E>> {
        if id < 1 || id > self.nodes.len() {
            return None; // boundary checks
        }
        let node = self.node(id);
        if node.start > self.elems[node.source].len() {
            return None; // don't return terminator nodes
        }
        let span = &self.elems[node.source][node.start..=node.end.min(self.end(node.source) - 1)];
        let edges = self
            .edges
            .get(&id)
            .map(|edges| {
                let mut elem_edges = HashMap::new();
                for (&token, &child) in edges {
                    match token {
                        Token::Element(elem) => {
                            elem_edges.insert(elem, child);
                        }
                        _ => (),
                    }
                }
                elem_edges
            })
            .unwrap_or_else(HashMap::new);
        let mut terminators = self
            .edges
            .get(&id)
            .map(|edges| {
                let mut terminators = Vec::new();
                for (&token, _) in edges {
                    match token {
                        Token::Terminator(source) => terminators.push(source),
                        _ => {}
                    }
                }
                terminators
            })
            .unwrap_or_else(Vec::new);

        if node.end >= self.elems[node.source].len() {
            terminators.push(node.source); // leaf node terminators are implicit
        }

        Some(NodeView {
            span,
            edges,
            terminators,
        })
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

    pub fn get_source(&self, source: usize) -> &[E] {
        &self.elems[source]
    }

    fn end(&self, source: usize) -> usize {
        self.elems[source].len()
    }

    fn add_link(&mut self, src: usize, dst: usize) {
        self.links.insert(src, dst);
    }

    pub fn insert(&mut self, elems: impl IntoIterator<Item = E>) -> usize {
        self.run_ukkonen(elems.into_iter())
    }

    fn run_ukkonen(&mut self, stream: impl Iterator<Item = E>) -> usize {
        let source = self.elems.len();
        self.elems.push(Vec::with_capacity(stream.size_hint().0));

        let mut node = ROOT;
        let mut start = 1;
        let mut i = 1;
        for elem in stream {
            self.elems[source].push(elem);
            (node, start) = self.update(source, node, start, i);
            (node, start) = self.canonize(source, node, start, i);
            i += 1;
        }
        // last update makes the tree explicit
        self.update(source, node, start, i);

        source
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

    pub fn contains(&self, pattern: &[E]) -> bool {
        self.find_one(pattern).is_some()
    }

    pub fn find_one(&self, pattern: &[E]) -> Option<(usize, usize)> {
        let n = pattern.len();
        let mut stack = Vec::from([(ROOT, 0)]);
        'outer: while let Some((node_id, i)) = stack.pop() {
            if let Some(child_id) = self.get_edge(node_id, pattern[i]) {
                let child = self.node(child_id);
                let m: usize = child.len().min(self.end(child.source) - child.start + 1);
                for j in 1..m {
                    if i + j == n {
                        // found a match
                        return Some((child.source, child.start + j - n - 1));
                    }

                    if self.get_elem(child.source, child.start + j) != pattern[i + j] {
                        continue 'outer;
                    }
                }
                if i + m == n {
                    // found a match
                    return Some((child.source, child.end.min(self.end(child.source)) - n));
                } else {
                    stack.push((child_id, i + m));
                }
            }
        }
        None
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
    fn test_contains_abab() {
        let result = GeneralizedSuffixTree::from(["abab".to_owned().chars()]);
        let mut pos = result.contains(&['a']);
        assert_eq!(pos, true);
        pos = result.contains(&['b', 'a']);
        assert_eq!(pos, true);
        pos = result.contains(&['b']);
        assert_eq!(pos, true);
        pos = result.contains(&['c']);
        assert_eq!(pos, false);
        pos = result.contains(&['b', 'c']);
        assert_eq!(pos, false);
        pos = result.contains(&['b', 'a', 'n', 'a', 'n', 'a']);
        assert_eq!(pos, false);
    }

    #[test]
    fn test_find_one_abab() {
        let result = GeneralizedSuffixTree::from(["abab".to_owned().chars()]);
        let mut pos = result.find_one(&['a']);
        assert_eq!(pos, Some((0, 0)));
        pos = result.find_one(&['b', 'a']);
        assert_eq!(pos, Some((0, 1)));
        pos = result.find_one(&['b']);
        assert_eq!(pos, Some((0, 1)));
        pos = result.find_one(&['c']);
        assert_eq!(pos, None);
        pos = result.find_one(&['b', 'c']);
        assert_eq!(pos, None);
        pos = result.find_one(&['b', 'a', 'n', 'a', 'n', 'a']);
        assert_eq!(pos, None);
    }

    #[test]
    fn test_find_one_banana() {
        let result = GeneralizedSuffixTree::from(["banana".to_owned().chars()]);
        let mut pos = result.find_one(&['b', 'a', 'n', 'a', 'n', 'a']);
        assert_eq!(pos, Some((0, 0)));
        pos = result.find_one(&['n', 'a']);
        assert_eq!(pos, Some((0, 2)));
        pos = result.find_one(&['a']);
        assert_eq!(pos, Some((0, 1)));
        pos = result.find_one(&['a', 'n', 'a']);
        assert_eq!(pos, Some((0, 1)));
    }

    #[test]
    fn test_find_one_multiple_sources() {
        let result =
            GeneralizedSuffixTree::from(["banana".to_owned().chars(), "anna".to_owned().chars()]);
        let mut pos = result.find_one(&['b', 'a', 'n', 'a', 'n', 'a']);
        assert_eq!(pos, Some((0, 0)));
        pos = result.find_one(&['a', 'n', 'n', 'a']);
        assert_eq!(pos, Some((1, 0)));
        pos = result.find_one(&['n', 'a']);
        assert_eq!(pos, Some((0, 2)));
        pos = result.find_one(&['a']);
        assert_eq!(pos, Some((0, 1)));
    }

    #[test]
    fn test_find_all_abab() {
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
    fn test_find_all_banana() {
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
    fn test_find_all_multiple_sources() {
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
