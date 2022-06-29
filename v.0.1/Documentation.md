El módulo "all_two_body_spin_ops(N, pauli_vec = list)", donde implemento los productos tensoriales para obtener los operadores de dos cuerpos, lo pensé como un árbol orientado. Por ejemplo:


           sx
        /  
       / 
    sx1 __ sy
   /   \
  /     \  sz
Sx         sx
  \     /  
   \   / 
    sx1 __ sy 
      \         sz1
       \      /  
           sz 
              \
               sz2
