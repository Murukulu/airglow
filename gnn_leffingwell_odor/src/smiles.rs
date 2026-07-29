use burn::optim::LBFGSConfig;
use purr::{
    feature::{Aliphatic, Aromatic, AtomKind, BondKind, BracketSymbol, Element},
    graph::{Atom, Bond, Builder},
};
use serde::{Deserialize, Deserializer, de};

fn deserialize_python_list<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    let items = inner
        .split(',')
        .map(|item| item.trim().trim_matches('\'').to_string())
        .filter(|s| !s.is_empty())
        .collect();
    Ok(items)
}

fn deserialize_smiles_to_features<'de, D>(deserializer: D) -> Result<Graph, D::Error>
where
    D: Deserializer<'de>,
{
    let mut builder = Builder::new();
    let mut graph = Graph {
        edges: Vec::new(),
        node_features: Vec::new(),
        edge_features: Vec::new(),
    };

    let smiles = String::deserialize(deserializer)?;
    purr::read::read(&smiles, &mut builder, None).unwrap();
    let atoms = builder
        .build()
        .map_err(|e| de::Error::custom(format!("{:?}", e)))?;
    for (i, atom) in atoms.iter().enumerate() {
        graph.node_features.push(feature_of_atom(&atom).to_vec());

        for bond in atom.bonds.iter() {
            graph.edges.push((i, bond.tid));
            graph.edge_features.push(feature_of_bond(bond).to_vec());
        }
    }
    Ok(graph)
}

#[derive(Debug, Deserialize)]
pub struct Chemical {
    chemical_name: String,
    #[serde(deserialize_with = "deserialize_smiles_to_features")]
    pub smiles: Graph,
    #[serde(rename = "database_id")]
    id: u32,
    cas: String,
    odor_data: String,
    #[serde(deserialize_with = "deserialize_python_list")]
    odor_labels_filtered: Vec<String>,
    #[serde(rename = "labels_train/test")]
    labels_train_test: f32,
    labels_cv0: f32,
    labels_cv1: f32,
    labels_cv2: f32,
    labels_cv3: f32,
    labels_cv4: f32,
}

#[derive(Debug)]
pub struct Graph {
    edges: Vec<(usize, usize)>,
    node_features: Vec<Vec<f32>>,
    edge_features: Vec<Vec<f32>>,
}

// Vocab: [C=0, N=1, O=2, S=3, F=4, Cl=5, Br=6, I=7, P=8, B=9, other=10]
// Output: 12-dim fixed vector — one-hot element (11) + is_aromatic (1)
// Subset of Chemprop atom features (omits degree, formal charge, chirality, Hs, hybridization, mass — require RDKit)
//
// TODO(saiputravu): For a lot of the features here, we are limited via purr and
// the member functions that it provides. We should swap to something like rdkit
// instead.
const ATOM_FEATURE_DIM: usize = 12;

fn feature_of_atom(a: &Atom) -> [f32; ATOM_FEATURE_DIM] {
    let (elem_idx, is_aromatic) = match &a.kind {
        AtomKind::Star => (10, false), // unknown
        AtomKind::Aliphatic(al) => (aliphatic_idx(al), false),
        AtomKind::Aromatic(ar) => (aromatic_idx(ar), true),
        AtomKind::Bracket { symbol, .. } => match symbol {
            BracketSymbol::Star => (10, false),
            BracketSymbol::Element(el) => (element_idx(el), false),
            // BracketAromatic converts to Element via Into<Element>
            BracketSymbol::Aromatic(ar) => {
                let el: Element = ar.into();
                (element_idx(&el), true)
            }
        },
    };

    // Setup one-hot.
    let mut feat = [0.0f32; ATOM_FEATURE_DIM];
    feat[elem_idx] = 1.0;
    // Add is_aromatic feature
    feat[11] = is_aromatic as u8 as f32;

    feat
}

// Vocab: [Single/Elided=0, Double=1, Triple=2, Quadruple=3, Aromatic=4]
// Output: 7-dim fixed vector — one-hot bond type (5) + is_aromatic (1) + is_directional (1)
// Chemprop uses (SINGLE, DOUBLE, TRIPLE, AROMATIC) + conjugated + in_ring + stereo — conjugated/in_ring require RDKit
const BOND_FEATURE_DIM: usize = 7;

fn feature_of_bond(b: &Bond) -> [f32; BOND_FEATURE_DIM] {
    let kind_idx = match b.kind {
        BondKind::Elided | BondKind::Single | BondKind::Up | BondKind::Down => 0, // single (Up/Down are directional singles)
        BondKind::Double => 1,
        BondKind::Triple => 2,
        BondKind::Quadruple => 3,
        BondKind::Aromatic => 4,
    };

    let mut feat = [0.0f32; BOND_FEATURE_DIM];
    feat[kind_idx] = 1.0;
    feat[5] = b.is_aromatic() as u8 as f32;
    feat[6] = b.is_directional() as u8 as f32;
    feat
}

fn aliphatic_idx(al: &Aliphatic) -> usize {
    match al {
        Aliphatic::C => 0,
        Aliphatic::N => 1,
        Aliphatic::O => 2,
        Aliphatic::S => 3,
        Aliphatic::F => 4,
        Aliphatic::Cl => 5,
        Aliphatic::Br => 6,
        Aliphatic::I => 7,
        Aliphatic::P => 8,
        Aliphatic::B => 9,
        Aliphatic::At | Aliphatic::Ts => 10, // other (rare halogens)
    }
}

fn aromatic_idx(ar: &Aromatic) -> usize {
    match ar {
        Aromatic::C => 0,
        Aromatic::N => 1,
        Aromatic::O => 2,
        Aromatic::S => 3,
        Aromatic::P => 8,
        Aromatic::B => 9,
    }
}

fn element_idx(el: &Element) -> usize {
    match el {
        Element::C => 0,
        Element::N => 1,
        Element::O => 2,
        Element::S => 3,
        Element::F => 4,
        Element::Cl => 5,
        Element::Br => 6,
        Element::I => 7,
        Element::P => 8,
        Element::B => 9,
        _ => 10, // other
    }
}
