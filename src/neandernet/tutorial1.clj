(ns neandernet.tutorial1
  (:require [uncomplicate.commons.core :refer [with-release let-release Releaseable release]]
            [uncomplicate.neanderthal
              [native :refer [dv dge]]
              [core :refer [mv! axpy! scal! transfer!]]
              [vect-math :refer [tanh! linear-frac!]]]
            [criterium.core :refer [quick-bench]])

  (:import clojure.lang.IFn))

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
    (activ-fn b (mv! w x h))))

(defn fully-connected [activ-fn in-dim out-dim]
  (let-release [w (dge out-dim in-dim)
                bias (dv out-dim)
                h (dv out-dim)]
    (->FullyConnectedInference w bias h activ-fn)))

(defn activ-sigmoid! [bias x]
  (axpy! -1.0 bias x)
  (linear-frac! 0.5 (tanh! (scal! 0.5 x)) 0.5))
(defn activ-tanh! [bias x]
  (tanh! (axpy! -1.0 bias x))  )


(defn test-fn [] 
  (with-release [x (dv 10000)
                 layer-1 (fully-connected activ-tanh! 10000 5000)
                 layer-2 (fully-connected activ-sigmoid! 5000 1000)
                 layer-3 (fully-connected activ-sigmoid! 1000 10)]
   ;; Call it like this:
    (quick-bench (layer-3 (layer-2 (layer-1 x))))
   ))
(def x (dv 10000))
(def layer-1 (fully-connected activ-tanh! 10000 5000))
(def layer-2 (fully-connected activ-sigmoid! 5000 1000))
(def layer-3 (fully-connected activ-sigmoid! 1000 10))
 
