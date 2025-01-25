
import numpy as np
import pandas as pd
from deepchem.feat.molecule_featurizers.mol_graph_conv_featurizer import _construct_atom_feature, \
    construct_hydrogen_bonding_info
from deepchem.feat.graph_features import atom_features
from spektral.utils import gcn_filter
from utils import zero_pad_graphs
from rdkit import Chem



class BaseFeaturizer(object):
    """Base class for chemical compound featurizers"""

    def __init__(self):
        self.featurizer_name = None

    def featurize_df(self, smiles_df, smiles_col, output_path=None):
        """
        Featurize a DataFrame of molecules represented as SMILES strings.

        Parameters
        ----------
        smiles_df: DataFrame
            DataFrame with the SMILES strings of the molecules that will be featurized
        smiles_col: str
            The name of the column containing the SMILES strings
        output_path: str
            Path to the CSV file where the featurized molecules will be saved.

        Returns
        -------
        featurized_df: DataFrame
            A pandas DataFrame containing the featurized molecules.
        """
        featurized_df = smiles_df[smiles_col].apply(self.featurize_molecule).apply(convert_numpy_to_list).apply(
            pd.Series)
        featurized_df = featurized_df.rename(columns=lambda x: self.featurizer_name + '_' + str(x + 1))

        if output_path is not None:
            featurized_df.to_csv(output_path, index=False)

        return featurized_df
    
    def featurize_molecule(self, smiles_string):
        """
        Featurize a single molecule.

        Parameters
        ----------
        smiles_string: str
            SMILES string representation of the molecule

        Raises
        -------
        NotImplementedError:
            This method is not implemented here since each subclass will have a different implementation.
        """
        raise NotImplementedError('Featurization method is not defined.')
    
    
    

class GraphFeaturizer(BaseFeaturizer):
    """Convert molecules represented as SMILES strings into molecular graphs. Each graph is defined using an adjacency
    matrix and the atom representations."""

    def __init__(self, zero_pad=True, max_num_atoms=96, normalize_adj_matrix=False, use_graphconv_featurization=False,
                 use_chirality=False, use_partial_charge=False):
        """
        Parameters
        ----------
        zero_pad: bool
            If True, zero-pads the graphs.
        max_num_atoms: int
            The maximum number of atoms allowed.
        normalize_adj_matrix: bool
            If True, the normalization described in the GCN (Kipf et al, 2017) paper will be applied.
        use_graphconv_featurization: bool
            If True, calculates the same atom features as DeepChem's ConvMolFeaturizer. Otherwise, it will calculate the
            atom features as defined for DeepChem's MolGraphConvFeaturizer.
        use_chirality: bool
            Whether to use chirality information or not.
        use_partial_charge: bool
            Whether to use partial charge data or not.
        """
        self.featurizer_name = 'graph'
        self.zero_pad = zero_pad
        self.max_num_atoms = max_num_atoms
        self.use_graphconv_featurization = use_graphconv_featurization
        self.use_partial_charge = use_partial_charge
        self.use_chirality = use_chirality
        self.normalize_adj = normalize_adj_matrix

    def featurize_df(self, smiles_df, smiles_col, output_path_node_features=None, output_path_adjacency_matrices=None):
        """
        Convert a DataFrame containing molecules represented as SMILES strings into the graph representation required by
        graph neural networks.

        Parameters
        ----------
        smiles_df: DataFrame
            DataFrame with the SMILES strings of the molecules that will be featurized
        smiles_col: str
            The name of the column containing the SMILES strings
        output_path_node_features
            Path to the file in which the atom features for all of the molecules will be saved.
        output_path_adjacency_matrices:
            Path to the file in which the adjacency matrices for all of the molecules will be saved.

        Returns
        -------
        node_features: array
            The atom features for all of the molecules in the DataFrame.
        adjacency_matrices: array
            The adjacency matrices for all of the molecules in the DataFrame.
        """
        node_features = []
        adjacency_matrices = []
        for smiles in smiles_df[smiles_col].tolist():
            nodes, adjacency_matrix = self.featurize_molecule(smiles)
            node_features.append(nodes)
            adjacency_matrices.append(adjacency_matrix)

        if self.zero_pad:
            # zero-padding here because otherwise I can't save as numpy arrays (this is what Spektral's BatchLoader does)
            node_features, adjacency_matrices = zero_pad_graphs(self.max_num_atoms, node_features,
                                                                              adjacency_matrices)

        if output_path_node_features is not None:
            np.save(output_path_node_features, node_features)

        if output_path_adjacency_matrices is not None:
            np.save(output_path_adjacency_matrices, adjacency_matrices)

        return node_features, adjacency_matrices
    def featurize_molecule(self, smiles_string):
        """
        Featurizes a single molecule for use with graph neural networks.

        Parameters
        ----------
        smiles_string: str
            SMILES string representation of the molecule.

        Returns
        -------
        node_features: array
            The atom features calculated for the molecule.
        adj_matrix: array
            The adjacency matrix for the molecule.
        """
        mol = Chem.MolFromSmiles(smiles_string)

        # Atom featurization is based on DeepChem code

        if self.use_graphconv_featurization:
            new_order = Chem.CanonicalRankAtoms(mol)
            mol = Chem.RenumberAtoms(mol, new_order)
            idx_nodes = [(a.GetIdx(), atom_features(a, use_chirality=False)) for a in mol.GetAtoms()]
            idx_nodes.sort()
            idx, nodes = list(zip(*idx_nodes))
            node_features = np.vstack(nodes)
        else:
            if self.use_partial_charge:
                try:
                    mol.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
                except:
                    from rdkit.Chem import AllChem
                    AllChem.ComputeGasteigerCharges(mol)
            # construct atom (node) features
            h_bond_infos = construct_hydrogen_bonding_info(mol)
            node_features = np.asarray(
                [_construct_atom_feature(atom, h_bond_infos, self.use_chirality, self.use_partial_charge) for atom in
                 mol.GetAtoms()], dtype=float)
            
            adj_matrix = Chem.GetAdjacencyMatrix(mol)
        if self.normalize_adj:
            adj_matrix = gcn_filter(adj_matrix)  # applies normalization described in Kipf et al, 2017 paper

        return node_features, adj_matrix
            