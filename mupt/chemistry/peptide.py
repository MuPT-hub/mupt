'''Reference for peptide residue substructures, FASTA codes, and CCD codes'''

__author__ = 'Timotej Bernat'
__email__ = 'timotej.bernat@colorado.edu'

import logging
LOGGER = logging.getLogger(__name__)

from dataclasses import dataclass, field

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import (
    MolFromSmiles,
    MolToSmiles,
    MolFromSmarts,
    MolFromFASTA,
)
from rdkit.Chem.rdmolops import (
    FragmentOnBonds,
    GetMolFrags,
    AddHs,
    SanitizeMol,
    SANITIZE_ALL,
)
from periodictable.fasta import AMINO_ACID_CODES

from .linkers import num_linkers


PEPTIDE_BOND_QUERY : Mol = MolFromSmarts('[$([CX3](=[OX1]))]-[$([NX3,NX4+](-C)(-C))]') # NOTE: final pair of carbons avoid overmatching asparagine AND under-matching proline
AMINE_QUERY : Mol = MolFromSmarts('[NH2,NH3+]')
CARBOXYL_QUERY : Mol = MolFromSmarts('C(=O)[OH,-O+]')

@dataclass(frozen=True)
class AminoAcidSubstructure:
    '''Encapsulation class for amino acid molecule file code and repeat unit substructure info'''
    name : str
    fasta : str
    ccd_code : str
    
    term_N_smiles : str = field(repr=False)
    middle_smiles : str = field(repr=False)
    term_O_smiles : str = field(repr=False)
    
    def term_N_fragment(self, removeHs : bool=False) -> Mol:
        '''Return amine/amide-terminated fragment as an RDKit Mol'''
        fragmol : Mol = MolFromSmiles(self.term_N_smiles, sanitize=removeHs)
        SanitizeMol(fragmol, sanitizeOps=SANITIZE_ALL)

        return fragmol

    def middle_fragment(self, removeHs : bool=False) -> Mol:
        '''Return middle fragment as an RDKit Mol'''
        fragmol : Mol = MolFromSmiles(self.middle_smiles, sanitize=removeHs)
        SanitizeMol(fragmol, sanitizeOps=SANITIZE_ALL)

        return fragmol

    def term_O_fragment(self, removeHs : bool=False) -> Mol:
        '''Return carboxyl-terminated fragment as an RDKit Mol'''
        fragmol : Mol = MolFromSmiles(self.term_O_smiles, sanitize=removeHs)
        SanitizeMol(fragmol, sanitizeOps=SANITIZE_ALL)

        return fragmol

def generate_amino_acid_substructures() -> set[AminoAcidSubstructure]:
    '''
    Procedurally generate amino acid terminal and middle fragment substructures
    '''
    aa_substructs : set[AminoAcidSubstructure] = set()
        
    for letter, ptabmol in AMINO_ACID_CODES.items():
        # DEV: opted for cleaving tripeptide (which is the smallest chain containing all unique linear fragments),
        # because peptide bond is much more forgiving to SMARTS query match for (no core-replacements needed)
        tripeptide = MolFromFASTA(3*letter) 
        if (tripeptide is None) or (tripeptide.GetNumAtoms() == 0):
            LOGGER.debug(f'RDKit has no amino acid registered to FASTA code "{letter}"; skipping')
            continue
        pdb_3_letter_code : str = tripeptide.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueName()
        
        # cleave along peptide bonds to produce head, middle, and tail AMINO_ACID fragments
        cleaved_tripeptide = FragmentOnBonds(
            tripeptide,
            bondIndices=[
                tripeptide.GetBondBetweenAtoms(*match).GetIdx()
                    for match in tripeptide.GetSubstructMatches(PEPTIDE_BOND_QUERY)
            ],
            dummyLabels=[(0,0),(0,0)],
        )
        cleaved_tripeptide.UpdatePropertyCache()
        cleaved_tripeptide = AddHs(cleaved_tripeptide)
        
        # extract fragments
        residue_fragments = GetMolFrags(
            cleaved_tripeptide,
            asMols=True,
            sanitizeFrags=False,
        )
        term_N_fragment, middle_fragment, term_O_fragment = residue_fragments # implicitly also enforces that there should be exactly 3 fragments
        
        # double-check that we have the terminal fragments labelled the right way around
        assert num_linkers(term_N_fragment) == 1
        assert num_linkers(middle_fragment) == 2
        assert num_linkers(term_O_fragment) == 1
        
        ## check functional groups
        # NOTE: can't directly check for presence of amine, since proline is a pig-headed, nonconformist idiot - also can't check for no carboxyls due to aspartic acid
        assert len(term_O_fragment.GetSubstructMatches(CARBOXYL_QUERY)) > 0
        
        aa_substruct = AminoAcidSubstructure(
            name=ptabmol.name.lower(),
            fasta=letter.upper(),
            ccd_code=pdb_3_letter_code.upper(),
            term_N_smiles=MolToSmiles(term_N_fragment),
            middle_smiles=MolToSmiles(middle_fragment),
            term_O_smiles=MolToSmiles(term_O_fragment),
        )
        aa_substructs.add(aa_substruct)
        
    return aa_substructs

AMINO_ACIDS_BY_NAME : dict[str, AminoAcidSubstructure] = dict()
AMINO_ACIDS_BY_FASTA : dict[str, AminoAcidSubstructure] = dict()
AMINO_ACIDS_BY_CCD : dict[str, AminoAcidSubstructure] = dict()
AMINO_ACIDS_BY_PDB_CODE = AMINO_ACIDS_BY_CCD # alias for convenience

for aa_substruct in generate_amino_acid_substructures():
    AMINO_ACIDS_BY_NAME[aa_substruct.name] = aa_substruct
    AMINO_ACIDS_BY_FASTA[aa_substruct.fasta] = aa_substruct
    AMINO_ACIDS_BY_CCD[aa_substruct.ccd_code] = aa_substruct