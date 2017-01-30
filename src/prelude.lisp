(lambda (x) x)
(lambda (x y) x y)
(lambda (x . rest) (+ x (car rest)))
