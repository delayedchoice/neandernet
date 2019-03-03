(ns neandernet.tutorial3
 (:require [uncomplicate.commons.core :refer [with-release let-release Releaseable release]]
           [uncomplicate.neanderthal
           [core :refer [axpy! scal! transfer! mm! rk! view-ge mv!]]
           [native :refer [dv dge]]
           [vect-math :refer [tanh! linear-frac!]]]) 

  (:import clojure.lang.IFn))
;sharing memory tutorial
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
  
  (invoke [_ x ones a]
    (activ-fn (rk! -1.0 b ones (mm! 1.0 w x 0.0 a)))))

(defn fully-connected [activ-fn in-dim out-dim]
  (let-release [w (dge out-dim in-dim)
                bias (dv out-dim)
                h (dv out-dim)]
    (->FullyConnectedInference w bias h activ-fn)))

(defn sigmoid! [x]
  (linear-frac! 0.5 (tanh! (scal! 0.5 x)) 0.5))

;broken
(let-release [temp-a (dv 8)]
  (with-release [x (dge 2 2 [0.3 0.9 0.3 0.9])
                 ones (dv 1 1)
                 layer-1 (fully-connected tanh! 2 4)
                 a-1 (view-ge temp-a 4 2)
                 layer-2 (fully-connected sigmoid! 4 1)
                 a-2 (view-ge temp-a 1 2)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (weights layer-1))
    (transfer! [0.7 0.2 1.1 2] (bias layer-1))
    (transfer! [0.75 0.15 0.22 0.33] (weights layer-2))
    (transfer! [0.3] (bias layer-2))
    (layer-2 (layer-1 x ones a-1) ones a-2)))

;fixed
(let-release [temp-odd (dv 8)
              temp-even (dv 2)]
  (with-release [x (dge 2 2 [0.3 0.9 0.3 0.9])
                 ones (dv 1 1)
                 layer-1 (fully-connected tanh! 2 4)
                 a-1 (view-ge temp-odd 4 2)
                 layer-2 (fully-connected sigmoid! 4 1)
                 a-2 (view-ge temp-even 1 2)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (weights layer-1))
    (transfer! [0.7 0.2 1.1 2] (bias layer-1))
    (transfer! [0.75 0.15 0.22 0.33] (weights layer-2))
    (transfer! [0.3] (bias layer-2))
    (layer-2 (layer-1 x ones a-1) ones a-2)))

