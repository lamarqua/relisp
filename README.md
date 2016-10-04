# ReLISP: (R)ust (e)xperiments in (LISP)

A toy / experiment lisp written in Rust. The goal of the exercise is both to learn Rust and its pitfalls of dynamic memory management and the implementation of a small Scheme-esque language with all the subtleties involved in interpreting a language.

The goal for the language itself is to be as minimal as possible, so that most of the things can be bootstrapped in ReLISP themselves. It also aims to have sensible "design decisions", like using lexical scoping for example. The goal for the interpreter is to remain efficient memory-wise and CPU-wise, in particular avoiding copying things as much as possible. A future goal is to move from an AST-walking interpreter to something more efficient. 

The parser is written in a recursive descent style with regex for lexing. 

Current state as of 03/10/2016: parsing OK, eval atoms OK, eval function / special forms KO.