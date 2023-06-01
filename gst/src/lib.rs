use std::collections::{BinaryHeap, HashMap, HashSet};
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
pub enum Token<E>
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
    pub fn is_elem(&self) -> bool {
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
    empty: HashMap<Token<E>, usize>, // workaround for lack of global generic
}

pub struct NodeView<'a, E>
where
    E: Copy + Eq + Hash + Debug,
{
    st: &'a GeneralizedSuffixTree<E>,
    id: usize,
    edges: &'a HashMap<Token<E>, usize>,
}

impl<'a, E> NodeView<'a, E>
where
    E: Default + Copy + Eq + Hash + Debug,
{
    pub fn id(&self) -> usize {
        self.id
    }
    pub fn source_id(&self) -> usize {
        self.st.node_unsafe(self.id).source
    }
    pub fn source(&self) -> &'a [E] {
        &self.st.elems[self.st.node_unsafe(self.id).source]
    }
    pub fn span_start(&self) -> usize {
        self.st.node_unsafe(self.id).start.saturating_sub(1) // edge case for root (0, 0)
    }
    pub fn span_end(&self) -> usize {
        let node = self.st.node_unsafe(self.id);
        if node.end == usize::MAX {
            self.st.elems[node.source].len()
        } else {
            node.end
        }
    }
    pub fn span(&'a self) -> &'a [E] {
        &self.source()[self.span_start()..self.span_end()]
    }
    pub fn len(&self) -> usize {
        self.span_end() - self.span_start()
    }
    pub fn edges(&self) -> &HashMap<Token<E>, usize> {
        &self.edges
    }
    pub fn link(&self) -> Option<NodeView<'a, E>> {
        self.st
            .links
            .get(&self.id)
            .map(|&id| self.st.node(id))
            .flatten()
    }
    pub fn link_id(&self) -> Option<usize> {
        self.st.links.get(&self.id).copied()
    }
    pub fn is_terminal(&self) -> bool {
        self.edges.is_empty()
    }
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

    pub fn root(&self) -> usize {
        ROOT
    }

    pub fn nodes_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn node(&self, id: usize) -> Option<NodeView<E>> {
        if id < 1 || id > self.nodes.len() {
            return None; // boundary checks
        }
        let edges = self.edges.get(&id).unwrap_or(&self.empty);

        Some(NodeView {
            st: &self,
            id,
            edges,
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

    fn edge_unsafe(&self, node: usize, key: Token<E>) -> usize {
        match node {
            0 => 1,
            n => self.edges[&n][&key],
        }
    }

    fn edge(&self, node: usize, elem: E) -> Option<usize> {
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

    fn node_unsafe(&self, node: usize) -> &Node {
        &self.nodes[node - 1]
    }

    fn node_unsafe_mut(&mut self, node: usize) -> &mut Node {
        &mut self.nodes[node - 1]
    }

    fn node_link_unsafe(&self, node: usize) -> usize {
        self.links[&node]
    }

    fn get_elem(&self, source: usize, pos: usize) -> Token<E> {
        if pos <= self.elems[source].len() {
            Token::Element(self.elems[source][pos - 1])
        } else {
            Token::Terminator(source)
        }
    }

    fn get_elem_unsafe(&self, source: usize, pos: usize) -> E {
        self.elems[source][pos - 1] // panics if index is outside of range
    }

    pub fn get_source(&self, source: usize) -> Option<&[E]> {
        self.elems.get(source).map(Vec::as_slice)
    }

    fn node_end(&self, source: usize) -> usize {
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
        let t_i = self.get_elem(source, i);
        let mut prev = ROOT;
        let (mut endpoint, mut curr) = self.test_and_split(source, node, start, i - 1, t_i);
        while !endpoint {
            let new = self.new_node(source, i, usize::MAX);
            self.add_edge(curr, t_i, new);
            if prev != ROOT {
                self.add_link(prev, curr);
            }
            prev = curr;
            (node, start) = self.canonize(source, self.node_link_unsafe(node), start, i - 1);
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
            let t_start = self.get_elem(source, start);
            let child = self.edge_unsafe(node, t_start);
            let t_mid = self.get_elem(
                self.node_unsafe(child).source,
                1 + self.node_unsafe(child).start + end - start,
            );
            if t == t_mid {
                (true, node)
            } else {
                let mid = self.new_node(
                    self.node_unsafe(child).source,
                    self.node_unsafe(child).start,
                    self.node_unsafe(child).start + end - start,
                );
                self.node_unsafe_mut(child).start += end - start + 1;
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
            let mut t_start = self.get_elem(source, start);
            let mut child = self.edge_unsafe(node, t_start);
            while end >= start
                && self.node_unsafe(child).end - self.node_unsafe(child).start <= end - start
            {
                start += self.node_unsafe(child).end - self.node_unsafe(child).start + 1;
                node = child;
                if start <= end {
                    t_start = self.get_elem(source, start);
                    child = self.edge_unsafe(node, t_start);
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
            if let Some(child_id) = self.edge(node_id, pattern[i]) {
                let child = self.node_unsafe(child_id);
                let m: usize = child
                    .len()
                    .min(self.node_end(child.source) - child.start + 1);
                for j in 1..m {
                    if i + j == n {
                        // found a match
                        return Some((child.source, child.start + j - n - 1));
                    }

                    if self.get_elem_unsafe(child.source, child.start + j) != pattern[i + j] {
                        continue 'outer;
                    }
                }
                if i + m == n {
                    // found a match
                    return Some((child.source, child.end.min(self.node_end(child.source)) - n));
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
            if let Some(child_id) = self.edge(node_id, pattern[i]) {
                let child = self.node_unsafe(child_id);
                let m: usize = child
                    .len()
                    .min(self.node_end(child.source) - child.start + 1);
                for j in 1..m {
                    if i + j == n {
                        // found a match
                        matches.extend(self.find_positions(child_id, i));
                        continue 'outer;
                    }

                    if self.get_elem_unsafe(child.source, child.start + j) != pattern[i + j] {
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
            let node = self.node_unsafe(node_id);
            let m = node.len().min(self.node_end(node.source) - node.start + 1);
            if let Some(edges) = self.edges.get(&node_id) {
                for (&token, &child) in edges {
                    match token {
                        Token::Element(_) => {
                            stack.push((child, depth + m));
                        }
                        Token::Terminator(source) => {
                            pos.push((source, self.node_end(source) - depth - m));
                        }
                    }
                }
            } else {
                pos.push((node.source, self.node_end(node.source) - depth - m));
            }
        }
        pos
    }

    fn find_lcp_recursive(
        &self,
        threshold: usize,
        min_length: usize,
        node_id: usize,
        length: usize,
        lcp: &mut BinaryHeap<(usize, usize)>,
    ) -> Option<HashSet<usize>> {
        let node = self.node(node_id).unwrap();
        let mut sources = if node.is_terminal() {
            HashSet::from([node.source_id()])
        } else {
            HashSet::new()
        };
        let mut propagate = false;
        for &child_id in node.edges().values() {
            if let Some(child_sources) =
                self.find_lcp_recursive(threshold, min_length, child_id, length + node.len(), lcp)
            {
                sources.extend(child_sources);
            } else {
                propagate = true;
            }
        }
        if propagate || length + node.len() < min_length {
            None
        } else if sources.len() >= threshold {
            lcp.push((length + node.len(), node_id));
            None
        } else {
            Some(sources)
        }
    }

    fn find_lcp(&self, threshold: usize, min_length: usize) -> BinaryHeap<(usize, usize)> {
        let mut lcp = BinaryHeap::new();
        self.find_lcp_recursive(threshold, min_length, self.root(), 0, &mut lcp);
        lcp
    }

    pub fn find_common(&self, threshold: usize, min_length: usize) -> Vec<&[E]> {
        let mut common = Vec::new();
        let mut to_visit = self.find_lcp(threshold, min_length);
        let mut visited = HashSet::new();
        while let Some((length, node_id)) = to_visit.pop() {
            if visited.contains(&node_id) {
                continue;
            }

            let mut node = self.node(node_id).unwrap();
            common.push(&node.source()[node.span_end() - length..node.span_end()]);

            while let Some(next) = node.link() {
                visited.insert(next.id());
                node = next;
            }
        }
        common
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
                let span = if self.node_unsafe(curr).start
                    > self.node_end(self.node_unsafe(curr).source)
                {
                    format!("T")
                } else {
                    format!(
                        "{:?}",
                        &self.elems[self.node_unsafe(curr).source][self.node_unsafe(curr).start - 1
                            ..self
                                .node_unsafe(curr)
                                .end
                                .min(self.node_end(self.node_unsafe(curr).source))]
                    )
                };
                write!(fmt, "{curr_prefix} {curr} {span}",)?;
                if !self.edges.contains_key(&curr) {
                    write!(fmt, ":{}", self.node_unsafe(curr).source)?;
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
    // fn test_foobar() {
    //     let gst = GeneralizedSuffixTree::from([
    //         "foo-123-bar".to_owned().chars(),
    //         "foo-456-bar".to_owned().chars(),
    //         "xyz-bar".to_owned().chars(),
    //         "foouvw".to_owned().chars(),
    //         "foopqr".to_owned().chars(),
    //         "foooqy".to_owned().chars(),
    //     ]);
    //     println!("{gst}");

    //     for common in gst.find_common(2, 2) {
    //         println!("{}", String::from_iter(common));
    //     }
    // }

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
