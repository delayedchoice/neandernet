(ns neandernet.tutorial2
  (:require [uncomplicate.commons.core :refer [with-release let-release Releaseable release]]
            [uncomplicate.neanderthal
              [native :refer [dv dge]]
              [core :refer [mv! axpy! scal! transfer! col mm! mm rk! entry! copy!]]
              [vect-math :refer [tanh! linear-frac!]]]
            [criterium.core :refer [quick-bench]])

  (:import clojure.lang.IFn))
;batch processing tutorial
;working with matrices of inputs instead of vectors
(defprotocol Parameters
  (weights [this])
  (bias [this]))

(deftype FullyConnectedInference [w b h activ-fn]
  Releaseable
  (release [_]
    (release w)
    (release b)
    (release h))
  Parameters
  (weights [this] w)
  (bias [this] b)
  IFn
  (invoke [_ x]
    (activ-fn (axpy! -1.0 b (mv! w x h))))
  (invoke [_ x ones a]
    (activ-fn (rk! -1.0 b ones (mm! 1.0 w x 0.0 a)))))

(defn fully-connected [activ-fn in-dim out-dim]
  (let-release [w (dge out-dim in-dim)
                bias (dv out-dim)
                h (dv out-dim)]
    (->FullyConnectedInference w bias h activ-fn)))

(defn sigmoid! [x]
  (linear-frac! 0.5 (tanh! (scal! 0.5 x)) 0.5))

(def x (dge 2 1 [0.3 0.9]))
(def w1 (dge 4 2 [0.3 0.6
                  0.1 2.0
                  0.9 3.7
                  0.0 1.0]
             {:layout :row}))
(def bias-vector (dv 0.7 0.2 1.1 2))

(let-release [a (dge 4 1)]
  (with-release [layer-1 (fully-connected sigmoid! 2 4)
                 ones (dv [1])]
    (copy! w1 (weights layer-1))
    (copy! bias-vector (bias layer-1))
    (layer-1 x ones a)))

;(with-release [x (dge 10000 10000)
;               ones (entry! (dv 10000) 1)
;               layer-1 (fully-connected tanh! 10000 5000)
;               a1 (dge 5000 10000)
;               layer-2 (fully-connected sigmoid! 5000 1000)
;               a2 (dge 1000 10000)
;               layer-3 (fully-connected sigmoid! 1000 10)
;               a3 (dge 10 10000)]
;
;  (time
;   (-> x
;       (layer-1 ones a1)
;       (layer-2 ones a2)
;       (layer-3 ones a3))))

 
