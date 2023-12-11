# Author: Felipe Reibnitz Willemann
import numpy as np
from typing import Tuple


class GFN1_xTB:
    def __init__(self, par_file: str) -> None:
        """Initialize."""
        self._read_parameters(par_file)
        return

    # =============================================================================
    # public methods
    def load_geometry(self, geom_file: str) -> bool:
        """Load geometry of the molecule from xyz file."""
        types = []
        coords = []
        with open(geom_file) as arq:
            try:
                size = int(arq.readline().split()[0])
            except IndexError:
                print("ERROR: Wrong molecule size format!")
                return False

            next(arq)  # skip title line
            i = 1
            for line in arq:
                line_split = line.split()
                if not line_split:
                    break  # blank line indicates end of file

                # check right number of coordinate elements per line
                elif len(line_split) != 4:
                    print(f"ERROR: Wrong coordinate format on line {i}!")
                    return False

                # check right coordinates number format
                try:
                    at_coord = np.array(line_split[1:]).astype(float)
                except ValueError:
                    print(f"ERROR: Wrong coordinate format on line {i}!")
                    return False

                types.append(line_split[0])
                coords.append(at_coord / self.b2a)
                i += 1

        # check correct molecule size
        if len(types) != size:
            print("ERROR: Wrong number of atoms!")
            return False

        # check correct atom types
        for i, type in enumerate(types):
            if type not in self.attypes.keys():
                print(f"ERROR: Incorrect type in atom {i+1}")
                return False

        # define molecule size, atom type labels and coordinates
        self.molsize = size
        self.moltypes = types
        self.molcoords = coords

        # define atom type numbers for molectule atoms (H: 0, C: 1, N: 2, O: 3)
        self.molattypes = [self.attypes[i] for i in self.moltypes]

        # define sequences of shell types and corresponding atom number of each orbital
        self.shelltypes, self.shellats = [], []
        for at in range(self.molsize):
            if self.moltypes[at] == "H":
                self.shelltypes += [1, 2]
                self.shellats += 2 * [at]
            else:
                self.shelltypes += [1, 3, 3, 3]
                self.shellats += 4 * [at]

        # calculate distance matrix
        self._calc_rAB()

        return True

    def scf(self, maxiter, out_file) -> None:
        """Perform Self Consistent Field calculation."""
        # calculate dispersion and repulsion energies and number of electrons
        edisp = self._calc_Edisp()
        erepul = self._calc_Erepul()
        ne = sum(
            self.H_shellprop[(at, shell)][0]
            for at in self.molattypes
            for shell in self.molshells[at]
        )

        # define squirt inverse overlap matrix for ortogonalizations
        overlap = self._calc_overlap_matrix()
        eval, evec = np.linalg.eig(overlap)
        s_sqrtinv = evec * 1 / np.sqrt(eval) @ evec.T

        # define initial charges, energy, zero-th order hamiltonian and first guess
        # for Fock matrix
        shell_charges0 = {
            (at, shell): 0
            for at in range(self.molsize)
            for shell in self.molshells[self.molattypes[at]]
        }
        eelec0 = 0
        h0 = self._calc_hamiltonian(overlap)
        fock = h0.copy()

        with open(out_file, "w+") as outfile:
            # write header and initial matrices
            outfile.write("---------------------------------------------\n")
            outfile.write("GFN1-xTB CALCULATION.\n")
            outfile.write("---------------------------------------------\n")
            outfile.write("\nMolecule coordinates (Bohr)\n")
            for at in range(self.molsize):
                outfile.write(
                    f"{self.moltypes[at]:<2}  "
                    + "{:>7.4f}  {:>7.4f}  {:>7.4f}\n".format(*self.molcoords[at])
                )
            outfile.write("\nOverlap (S)\n")
            np.savetxt(outfile, overlap, fmt="%9.6f")
            outfile.write("\nZero-th order Hamiltonian\n")
            np.savetxt(outfile, h0, fmt="%9.6f")

            # start SCF loop
            i = 1
            while i < maxiter:
                # ortogonalize and diagonalize the Fock matrix
                fockort = s_sqrtinv.T @ fock @ s_sqrtinv
                eval, evec = np.linalg.eig(fockort)
                order = np.argsort(eval)

                # calculate ordered coefficient and density matrix and charges
                c = s_sqrtinv @ evec[:, order]
                density = self._calc_density(c=c, ne=ne)
                shell_charges, atom_charges = self._calc_charges(overlap, density)

                # write first density ans start SCF cycle
                if i == 1:
                    outfile.write("\nDensity first guess\n")
                    np.savetxt(outfile, density, fmt="%9.6f")
                    outfile.write("\n---------------------------------------------\n")
                    outfile.write("STARTING SCF CYCLES.\n")
                    outfile.write("---------------------------------------------\n")
                    outfile.write(
                        "\nStep       E_elec       E_1          E_2          E_3       Damping\n"
                    )

                # calculate electronic energies
                e1 = self._calc_E1(h0, density)
                e2 = self._calc_E2(shell_charges)
                e3 = self._calc_E3(atom_charges)
                eelec = e1 + e2 + e3

                # charge difference and damping
                charge_diff = abs(
                    np.array(list(shell_charges.values()))
                    - np.array(list(shell_charges0.values()))
                )
                damping: bool = not max(charge_diff) < 1e-3

                # print step results
                outfile.write(
                    f" {i:>2}    {eelec:>11.6f}  {e1:>11.6f}  {e2:>11.6f}  {e3:>11.6f}     {['no', 'yes'][damping]}\n"
                )

                # check for convergence, write final results
                if abs(eelec - eelec0) < 1e-7:
                    outfile.write("\n---------------------------------------------\n")
                    outfile.write(f"CONVERGENCE REACHED ON STEP {i}.\n")
                    outfile.write("---------------------------------------------\n")
                    outfile.write("\nFinal atomic charges:\n")
                    for at in range(self.molsize):
                        outfile.write(
                            f"{self.moltypes[at]:<2}  {atom_charges[at]:>10.6f}\n"
                        )
                    outfile.write("Final orbital energies:\n")
                    np.savetxt(outfile, eval[order], fmt="%9.6f", newline=" ")
                    outfile.write("\nFinal orbital coefficients:\n")
                    np.savetxt(outfile, c.T, fmt="%9.6f")
                    outfile.write(
                        f"Final electronic energy (E_1, E_2, E_3): {eelec:.6f} ({e1:.6f}, {e2:.6f}, {e3:.6f})\n"
                    )
                    outfile.write(
                        f"Repulsion and dispersion energies: {erepul:.8f}, {edisp:.8f}\n"
                    )
                    outfile.write(
                        f"Final total energy: {eelec + erepul + edisp:.8f} Hartree"
                    )
                    outfile.write(f" ({(eelec + erepul + edisp)/self.eV2h:.8f} eV)")
                    print("INFO: Calculation finished normally.")
                    return

                # check for big changes in charges (damping method)
                elif damping:
                    damped_charges = self._damp_charges(
                        shell_charges0, shell_charges, 0.4
                    )
                    fock = self._calc_fock_matrix(
                        overlap, h0, damped_charges[0], damped_charges[1]
                    )

                # otherwise calculate next Fock matrix normally
                else:
                    fock = self._calc_fock_matrix(
                        overlap, h0, shell_charges, atom_charges
                    )

                # define previous step charges and energies
                shell_charges0 = shell_charges
                eelec0 = eelec
                i += 1

        print(f"WARNING: Calculation did not converge within {maxiter} iterations!")
        return

    # =============================================================================
    # private methods
    def _read_parameters(self, file_name: str) -> None:
        """Read values from paramaters file."""
        # get all parameter lines splitted
        p = []
        with open(file_name) as arq:
            for line in arq.readlines():
                line_split = line.split()

                if (len(line_split) == 0) or ("#" in line_split[0]):
                    continue
                elif len(line_split) == 1:
                    p.append(line_split[0])
                else:
                    p.append(line_split)

        # global parameters
        self.b2a = float(p[0])
        self.eV2h = 1 / float(p[1])
        self.nattypes = int(p[2])
        self.attypes = {at: i for i, at in enumerate(p[3])}

        # zeroth-order parameters
        self.kf = float(p[4])
        self.alpha = list(map(float, p[5]))
        self.zeff = list(map(float, p[6]))

        # dispersion parameters
        self.an = list(map(float, p[7][:2]))
        self.sn = list(map(float, p[7][2:4]))
        self.kCN = float(p[7][4])
        self.kL = float(p[7][5])
        self.QA = list(map(float, p[8]))
        self.covradii = 4 * np.array(p[9]).astype(float) / (3 * self.b2a)
        self.nrefCN = list(map(int, p[11]))
        self.refCN = {
            list(self.attypes.keys())[i]: list(map(float, p[12 + i]))
            for i in range(self.nattypes)
        }

        # reference C6 parameters
        self.refC6 = {}
        init = 17
        for i in range(self.nattypes):
            for j in range(i, self.nattypes):
                self.refC6[(i, j)] = [
                    list(map(float, pel)) for pel in p[init : init + self.nrefCN[i]]
                ]
                init += 1 + self.nrefCN[i]

        # basis functions
        self.basis = {}
        init = 60
        for i in range(2 * self.nattypes):
            self.basis[(int(p[init][0]) - 1, int(p[init][1]))] = [
                list(map(float, pel)) for pel in p[init + 1 : init + 3]
            ]
            init += 3

        # Hamiltonian parameters
        self.H_kAB = list(map(float, p[84]))
        self.H_nkll = int(p[85])
        self.H_kll = np.zeros((3, 3))
        k = 0
        for i in range(3):
            for j in range(i, 3):
                self.H_kll[i, j] = float(p[86 + k][2])
                self.H_kll[j, i] = self.H_kll[i, j]
                k += 1
        self.H_elecneg = list(map(float, p[92]))
        self.H_kEN = float(p[93])
        self.H_covradii = np.array(p[94]).astype(float) / self.b2a
        self.H_chargedev = list(map(float, p[95]))
        self.H_shellprop = {
            (int(p[96 + i][0]) - 1, int(p[96 + i][1])): [int(p[96 + i][2])]
            + list(map(float, p[96 + i][3:]))
            for i in range(2 * self.nattypes)
        }
        self.H_kABscall = list(map(float, p[104]))

        # define molecular shell types for each atom type
        self.molshells = np.array([[1, 2], [1, 3], [1, 3], [1, 3]], dtype=int)

    def _calc_rAB(self) -> None:
        """Distance matrix of the molecule."""
        self.rAB = np.zeros((self.molsize, self.molsize))
        for i in range(self.molsize):
            for j in range(i):
                diff = self.molcoords[i] - self.molcoords[j]
                self.rAB[i, j] = np.linalg.norm(diff)
                self.rAB[j, i] = self.rAB[i, j]

    # -----------------------------------------------------------------------------
    ## repulsion and dispersive energy methods
    def _calc_Erepul(self) -> float:
        """Zero-th order repulsive energy."""
        Erepul = 0
        for i in range(self.molsize):
            a = self.molattypes[i]
            for j in range(i):
                b = self.molattypes[j]
                term = (
                    -np.sqrt(self.alpha[a] * self.alpha[b]) * self.rAB[i, j] ** self.kf
                )
                Erepul += self.zeff[a] * self.zeff[b] * np.exp(term) / self.rAB[i, j]

        return Erepul

    def _calc_Edisp(self) -> float:
        """Zero-th order dispersive energy."""
        Edisp = 0
        c6 = self._calc_C6()
        c8 = self._calc_C8(c6=c6)
        rAB0 = (9 * np.outer(self.QA, self.QA)) ** (1 / 4)

        for i in range(self.molsize):
            a = self.molattypes[i]
            for j in range(i):
                b = self.molattypes[j]
                for k, n in enumerate([6, 8]):
                    fdamp = (
                        self.rAB[i, j] ** n
                        + (self.an[0] * rAB0[a, b] + self.an[1]) ** n
                    )  # damping function divided by atom pair distance ** n
                    Edisp += -self.sn[k] * [c6, c8][k][i, j] / fdamp

        return Edisp

    def _calc_weights(self) -> np.array:
        """Weights of coordination numbers for every atom in the molecule."""
        # first calculate coordination numbers from reference values
        self.cn = np.zeros(self.molsize)
        for i in range(self.molsize):
            n = self.molattypes[i]
            for j in range(self.molsize):
                # skip diagonal terms
                if i == j:
                    continue

                m = self.molattypes[j]
                rterm = (self.covradii[n] + self.covradii[m]) / self.rAB[i, j] - 1
                self.cn[i] += 1 / (1 + np.exp(-self.kCN * rterm))

        return [
            np.exp(-self.kL * np.square(self.cn[i] - self.refCN[a]))
            for i, a in enumerate(self.moltypes)
        ]

    def _calc_C6(self) -> np.array:
        """6th order coefficient for every atom pair in the molecule."""
        ln = self._calc_weights()
        c6 = np.zeros((self.molsize, self.molsize))
        for i in range(self.molsize):
            a = self.molattypes[i]
            na = len(ln[i])

            for j in range(i):
                b = self.molattypes[j]
                nb = len(ln[j])
                refC6 = self.refC6[(a, b)]

                numerator = sum(
                    sum(refC6[n][m] * ln[i][n] * ln[j][m] for n in range(na))
                    for m in range(nb)
                )
                denominator = sum(
                    sum(ln[i][n] * ln[j][m] for n in range(na)) for m in range(nb)
                )

                c6[i, j] = numerator / denominator
                c6[j, i] = c6[i, j]

        return c6

    def _calc_C8(self, c6: np.array) -> np.array:
        """8th order coefficient for every atom in the molecule."""
        c8 = np.zeros((self.molsize, self.molsize))
        for i in range(self.molsize):
            a = self.molattypes[i]
            for j in range(i):
                b = self.molattypes[j]
                term = np.sqrt(self.QA[a] * self.QA[b])
                c8[i, j] = 3 * c6[i, j] * term
                c8[j, i] = c8[i, j]

        return c8

    # -----------------------------------------------------------------------------
    ## basis function methods
    def _calc_overlap_matrix(self) -> np.array:
        """Overlap matrix between atomic basis functions."""
        sdim = len(self.shelltypes)
        overlap = np.zeros((sdim, sdim))

        # define p orbital index (0 = x, 1 = y, 2 = z)
        pidx = self.shelltypes.copy()
        for i in range(sdim):
            if pidx[i] == 3:
                pidx[i] = 0
                pidx[i + 1] = 1
                pidx[i + 2] = 2

        for mu in range(sdim):
            muat = self.shellats[mu]
            mushell = self.shelltypes[mu]

            for nu in range(mu + 1):
                nuat = self.shellats[nu]
                nushell = self.shelltypes[nu]

                overlap[mu, nu] = self._calc_overlap_el(
                    muat, nuat, mushell, nushell, pidx[mu], pidx[nu]
                )
                overlap[nu, mu] = overlap[mu, nu]

        return overlap

    def _calc_overlap_el(
        self, muat: int, nuat: int, mushell: int, nushell: int, mupidx: int, nupidx: int
    ) -> float:
        """Overlap matrix element."""
        muattype = self.molattypes[muat]
        nuattype = self.molattypes[nuat]
        mucoefs = self.basis[(muattype, mushell)]
        nucoefs = self.basis[(nuattype, nushell)]
        mur = self.molcoords[muat]
        nur = self.molcoords[nuat]

        # initiate element value and loops over coefficients
        el = 0
        for muxi, mud in zip(*mucoefs):
            for nuxi, nud in zip(*nucoefs):
                # ovelap parameters
                xi = muxi + nuxi
                chi = muxi * nuxi / xi
                rp = (muxi * mur + nuxi * nur) / xi

                # calculate (0||0) and keep in final s value
                s00 = np.exp(-chi * np.square(self.rAB[muat, nuat])) * (np.pi / xi) ** (
                    3 / 2
                )
                sfinal = s00

                # account for mu p orbital
                if mushell == 3:
                    sfinal *= rp[mupidx] - mur[mupidx]

                # account for nu p orbital
                if nushell == 3:
                    sfinal *= rp[nupidx] - nur[nupidx]

                # account for equal mu and nu p orbital index (x, y or z)
                if (nushell == 3) and (mushell == 3) and (mupidx == nupidx):
                    sfinal += s00 / (2 * xi)

                el += mud * nud * sfinal

        return el

    # -----------------------------------------------------------------------------
    ## hamiltonian and first order energy methods
    def _calc_hamiltonian(self, overlap: np.array) -> np.array:
        """Zero-th order Hamiltonian matrix."""
        sdim = len(self.shellats)
        hamiltonian = np.zeros((sdim, sdim))
        for mu in range(sdim):
            muat = self.shellats[mu]
            muattype = self.molattypes[muat]
            mushell = self.shelltypes[mu]

            # effective mu atomic energy level
            muh = self.H_shellprop[(muattype, mushell)][2] * self.eV2h
            muh *= 1 + self.H_kABscall[mushell - 1] * self.cn[muat]

            for nu in range(mu + 1):
                nuat = self.shellats[nu]
                nuattype = self.molattypes[nuat]
                nushell = self.shelltypes[nu]

                # effective nu atomic energy level
                nuh = self.H_shellprop[(nuattype, nushell)][2] * self.eV2h
                nuh *= 1 + self.H_kABscall[nushell - 1] * self.cn[nuat]

                # check same shells on same atom condition
                if muat == nuat:
                    if mu == nu:
                        hamiltonian[mu, nu] = muh
                    continue

                pi = self._calc_pi_function(muat, nuat, mushell, nushell)
                hel = (muh + nuh) * self.H_kll[mushell - 1, nushell - 1]
                hel *= overlap[mu, nu] * pi / 2

                # account fpr non s' orbital off diagonal scaling factor
                if mushell != 2 and nushell != 2:
                    hel *= 1 + self.H_kEN * np.square(
                        self.H_elecneg[muattype] - self.H_elecneg[nuattype]
                    )

                # check scaling factor for HHss and HNss/sp
                if muattype == 0 and nuattype == 0 and mushell == 1 and nushell == 1:
                    hel *= self.H_kAB[0]

                elif (muattype == 0 and nuattype == 2 and mushell == 1) or (
                    muattype == 2 and nuattype == 0 and nushell == 1
                ):
                    hel *= self.H_kAB[1]

                hamiltonian[mu, nu] = hel
                hamiltonian[nu, mu] = hel

        return hamiltonian

    def _calc_pi_function(
        self, muat: int, nuat: int, mushell: int, nushell: int
    ) -> float:
        """Distance and shell dependent polynomial scaling function."""
        muattype = self.molattypes[muat]
        nuattype = self.molattypes[nuat]

        # radius term
        rterm = np.sqrt(
            self.rAB[muat, nuat]
            / (self.H_covradii[muattype] + self.H_covradii[nuattype])
        )

        mukpoly = self.H_shellprop[(muattype, mushell)][3]
        nukpoly = self.H_shellprop[(nuattype, nushell)][3]
        return (1 + mukpoly * rterm) * (1 + nukpoly * rterm)

    def _calc_E1(self, hamiltonian: np.array, density: np.array) -> float:
        """First order electric energy."""
        sdim = len(self.shellats)
        e1 = 0
        for i in range(sdim):
            for j in range(sdim):
                e1 += hamiltonian[i, j] * density[i, j]

        return e1

    # -----------------------------------------------------------------------------
    ## second and third order energy methods
    def _calc_shell_charge(
        self, at: int, shell: int, overlap: np.array, density: np.array
    ) -> float:
        """Total charge in selected atomic shell."""
        attype = self.molattypes[at]
        sdim = len(self.shellats)
        charge = 0

        for mu in range(sdim):
            # check if in same atom shell
            if self.shellats[mu] != at or self.shelltypes[mu] != shell:
                continue

            for nu in range(sdim):
                charge -= overlap[mu, nu] * density[mu, nu]

        return charge + self.H_shellprop[(attype, shell)][0]

    def _calc_charges(
        self, overlap: np.array, density: np.array
    ) -> Tuple[np.array, np.array]:
        """Charges of shells and atoms for given density."""
        shell_charges = {}
        atom_charges = np.zeros(self.molsize)
        for at in range(self.molsize):
            attype = self.molattypes[at]
            for shell in self.molshells[attype]:
                shell_charges[(at, shell)] = self._calc_shell_charge(
                    at, shell, overlap, density
                )
                atom_charges[at] += shell_charges[(at, shell)]

        return shell_charges, atom_charges

    def _calc_gamma(self, a: int, b: int, ashell: int, bshell: int) -> float:
        """Couloumb interaction term Gamma between selected atomic shells."""
        atype = self.molattypes[a]
        btype = self.molattypes[b]
        etaa = self.H_shellprop[(atype, ashell)][1]
        etab = self.H_shellprop[(btype, bshell)][1]

        etainv = (1 / etaa + 1 / etab) / 2
        return 1 / np.sqrt(self.rAB[a, b] ** 2 + etainv**2)

    def _calc_E2(self, shell_charges) -> float:
        """Second order electric energy."""
        e2 = 0
        for a in range(self.molsize):
            atype = self.molattypes[a]
            for b in range(self.molsize):
                btype = self.molattypes[b]
                for ashell in self.molshells[atype]:
                    acharge = shell_charges[(a, ashell)]
                    for bshell in self.molshells[btype]:
                        bcharge = shell_charges[(b, bshell)]
                        gamma = self._calc_gamma(a, b, ashell, bshell)

                        e2 += acharge * bcharge * gamma

        return e2 / 2

    def _calc_E3(self, atom_charges: np.array) -> float:
        """Third order electric energy."""
        return (1 / 3) * sum(
            self.H_chargedev[self.molattypes[at]] * atom_charges[at] ** 3
            for at in range(self.molsize)
        )

    # -----------------------------------------------------------------------------
    ## SCF methods
    def _calc_sls(self, a: int, ashell: int, shell_charges: np.array) -> float:
        """Shell level shift."""
        sls = 0
        for b in range(self.molsize):
            btype = self.molattypes[b]
            for bshell in self.molshells[btype]:
                gamma = self._calc_gamma(a, b, ashell, bshell)
                bcharge = shell_charges[(b, bshell)]
                sls += gamma * bcharge

        return sls

    def _calc_als(self, at: int, atom_charges: np.array) -> float:
        """Atom level shift."""
        return self.H_chargedev[self.molattypes[at]] * atom_charges[at] ** 2

    def _calc_fock_matrix(
        self,
        overlap: np.array,
        hamiltonian: np.array,
        shell_charges: np.array,
        atom_charges: np.array,
    ) -> np.array:
        """Fock matrix."""
        sdim = len(self.shellats)
        fock = np.zeros((sdim, sdim))

        for mu in range(sdim):
            muat = self.shellats[mu]
            mushell = self.shelltypes[mu]
            musls = self._calc_sls(muat, mushell, shell_charges)
            muals = self._calc_als(muat, atom_charges)

            for nu in range(sdim):
                nuat = self.shellats[nu]
                nushell = self.shelltypes[nu]
                nusls = self._calc_sls(nuat, nushell, shell_charges)
                nuals = self._calc_als(nuat, atom_charges)

                fock[mu, nu] = (
                    hamiltonian[mu, nu]
                    - overlap[mu, nu] * (musls + nusls + muals + nuals) / 2
                )

        return fock

    def _calc_density(self, c: np.array, ne: int) -> np.array:
        """Density matrix."""
        sdim = len(self.shellats)
        p = np.zeros((sdim, sdim))
        for mu in range(sdim):
            for nu in range(sdim):
                p[mu, nu] = 2 * sum(c[mu, k] * c[nu, k] for k in range(int(ne / 2)))

        return p

    def _damp_charges(
        self, shell_charges0: np.array, shell_charges: np.array, lam: float
    ) -> np.array:
        """Damped charges method."""
        atom_charges = np.zeros(self.molsize)
        for at in range(self.molsize):
            attype = self.molattypes[at]
            for shell in self.molshells[attype]:
                shell_charges[(at, shell)] = (
                    lam * shell_charges[(at, shell)]
                    + (1 - lam) * shell_charges0[(at, shell)]
                )
                atom_charges[at] += shell_charges[(at, shell)]

        return shell_charges, atom_charges
