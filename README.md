# dwr-linear

### Description
The repository contains a code to estimate (a posteriori) the error of Finite Element Method applied to a human tongue model with fiber activation. From that estimation, the mesh used can be refined adaptively, which leads to higher convergence rate compared to those obtained by uniform refinement technique.

Running the code will give the results presented in Section 3.1 of the paper:

Quantifying discretization errors for soft-tissue simulation in computer assisted surgery: a preliminary study, Michel Duprez, Stéphane P.A. Bordas, Marek Bucki, Huu Phuoc Bui, Franz Chouly, Vanessa Lleras, Claudio Lobos, Alexei Lozinski, Pierre-Yves Rohan, Satyendra Tomar. https://arxiv.org/abs/1806.06944

### Instructions
1. Install FEniCS for your platform following the instruction at
   [fenicsproject.org](https://fenicsproject.org).
   
2. Clone this repository:

      ```      
        git clone https://huuphuocbui@bitbucket.org/huuphuocbui/dwr-linear.git
      ```

3. You can edit the dwr-linear.py to specify the uniform or adaptive refinement scheme to be used (line 387)

4. In a terminal, enter the directory of the repository, and launch the code using
   the command:

      ```      
      cd dwr-linear      
      ```
      
      ```      
      python3 dwr-linear.py
      ```

5. The output files, including the text files containing data for Figure 5 in the paper,  will be generated in the current directory.


### Issues and Support

For support or questions please email [michel.duprez@math.cnrs.fr](mailto:michel.duprez@math.cnrs.fr ), [huu-phuoc.bui@alumni.unistra.fr](mailto:huu-phuoc.bui@alumni.unistra.fr).

### Authors

Michel Duprez, Laboratoire Jacques-Louis Lions, Sorbonne Université, France.

Huu Phuoc Bui, Université de Franche-Comté, France.

### License 

dwr-linear is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with dwr-linear.  If not, see <http://www.gnu.org/licenses/>.