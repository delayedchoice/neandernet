(ns neandernet.gpu-tutorial
  (:require [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.clojurecl.info :refer :all])

  (:require [uncomplicate.commons.core :refer [with-release  let-release Releaseable release]]
            [uncomplicate.clojurecl
             [core :refer [sort-by-cl-version set-context! set-queue! with-platform platforms with-context context with-queue
                           sort-by-cl-version devices with-default-1 command-queue-1
                           *context* *command-queue* *platform* finish!
                           set-default-1! release-context!]]]
            [uncomplicate.neanderthal
             [core :refer [asum axpy! scal! transfer! transfer mm! rk! view-ge vctr ge entry!]]
             [native :refer [native-double native-float dv dge]]  
             [vect-math :refer [tanh! linear-frac!]]
             [opencl :refer [opencl-float clv with-default-engine set-engine!]]])  
  (:import clojure.lang.IFn)  )

(defprotocol Parameters
  (weights [this])
  (bias [this]))

(deftype FullyConnectedInference [w b activ-fn]
  Releaseable
  (release [_]
    (release w)
    (release b))
  Parameters
  (weights [this] w)
  (bias [this] b)
  IFn
  (invoke [_ x ones a]
    (activ-fn (rk! -1.0 b ones (mm! 1.0 w x 0.0 a)))))

(defn sigmoid! [x]
  (linear-frac! 0.5 (tanh! (scal! 0.5 x)) 0.5))

(defn fully-connected [factory activ-fn in-dim out-dim]
  (let-release [w (ge factory out-dim in-dim)
                bias (vctr factory out-dim)]
    (->FullyConnectedInference w bias activ-fn)))
;a general function that takes the factory (hardware implemenetaion (cpu or gpu))
(defn this-particular-network [factory]
  (with-release [x (ge factory 2 2 [0.3 0.9 0.3 0.9])
                 ones (vctr factory 1 1)
                 layer-1 (fully-connected factory tanh! 2 4)
                 a-1 (ge factory 4 2)
                 layer-2 (fully-connected factory sigmoid! 4 1)
                 a-2 (ge factory 1 2)]
    (transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (weights layer-1))
    (transfer! [0.7 0.2 1.1 2] (bias layer-1))
    (transfer! [0.75 0.15 0.22 0.33] (weights layer-2))
    (transfer! [0.3] (bias layer-2))
    (transfer (layer-2 (layer-1 x ones a-1) ones a-2))))

;(set-default-1!)
;(set-engine!)

;(release-context!)


;(with-default-1
;  (with-default-engine
;    (with-release [opencl-factory (opencl-float *context* *command-queue*)]
;      (this-particular-network opencl-factory))))

(with-platform (first (platforms))
     (let [dev (second (sort-by-cl-version (devices)))]
       (with-context (context [dev])
         (with-queue (command-queue-1 dev)
          (with-release [factory (opencl-float *context* *command-queue*)]
            (with-release [x (ge factory 10000 10000)
                    ones (entry! (vctr factory 10000) 1)
                    layer-1 (fully-connected factory tanh! 10000 5000)
                    a1 (ge factory 5000 10000)
                    layer-2 (fully-connected factory sigmoid! 5000 1000)
                    a2 (ge factory 1000 10000)
                    layer-3 (fully-connected factory sigmoid! 1000 10)
                    a3 (ge factory 10 10000)]
       ;(layer-1 x ones a1) ;; The first time a BLAS operation is used in OpenCL might incur initialization cost.
         (do
           (transfer (layer-3 (layer-2 (layer-1 x ones a1) ones a2) ones a3))
           ))) ))))

;(with-default-1
;  (with-default-engine
;    (with-release [factory (opencl-float *context* *command-queue*)]
;     (with-release [x (ge factory 10000 10000)
;                    ones (entry! (vctr factory 10000) 1)
;                    layer-1 (fully-connected factory tanh! 10000 5000)
;                    a1 (ge factory 5000 10000)
;                    layer-2 (fully-connected factory sigmoid! 5000 1000)
;                    a2 (ge factory 1000 10000)
;                    layer-3 (fully-connected factory sigmoid! 1000 10)
;                    a3 (ge factory 10 10000)]
;       ;(layer-1 x ones a1) ;; The first time a BLAS operation is used in OpenCL might incur initialization cost.
;         (do
;           (transfer (layer-3 (layer-2 (layer-1 x ones a1) ones a2) ones a3))
;           )))))
;("Intel(R) Core(TM) i5-7Y54 CPU @ 1.20GHz" "Intel(R) HD Graphics 615")
;("Intel(R) Core(TM) i5-7Y54 CPU @ 1.20GHz" "Intel(R) HD Graphics 615")
;(map name-info (devices (first (platforms))))
(def intel-platform (first (platforms)))
(def my-intel-gpu (second (devices intel-platform)))
(def ctx (context [my-intel-gpu]))
(def cq (command-queue-1 ctx my-intel-gpu))
;(set-context! ctx)
;(set-queue! cq)
;(def device-name "Intel(R) HD Graphics 615")
;(def devices [ (second (devices (first (platforms)))) ])
