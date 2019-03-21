(ns neandernet.tutorial-8
 (:require [uncomplicate.commons.core :refer [with-release let-release Releaseable release]]
           [uncomplicate.clojurecl.core :refer [with-context with-queue devices platforms with-platform finish! context command-queue-1]]
           [uncomplicate.neanderthal
             [core :refer [mrows dim raw view copy! scal! transfer!  transfer mm! rk! vctr ge entry!]]
             [native :refer [native-float]]
             [vect-math :refer [tanh! linear-frac!]]
             [opencl :refer [with-default-engine opencl-float]]])
  (:require [neandernet.util :refer [with-open-cl]])
  (:import clojure.lang.IFn))

(defprotocol Parameters
  (weights [this])
  (bias [this]))

(defprotocol ActivationProvider
  (activation-fn [this]))

(deftype FullyConnectedInference [w b activ-fn]
  Releaseable
  (release [_]
    (release w)
    (release b))
  Parameters
  (weights [_] w)
  (bias [_] b)
  ActivationProvider
  (activation-fn [_] activ-fn)
  IFn
  (invoke [_ x ones a]
    (activ-fn (rk! -1.0 b ones (mm! 1.0 w x 0.0 a)))))

(defn fully-connected [factory activ-fn in-dim out-dim]
  (let-release [w (ge factory out-dim in-dim)
                bias (vctr factory out-dim)]
    (->FullyConnectedInference w bias activ-fn)))

;training
(defprotocol Backprop
  (forward [this])
  (backward [this]))

(defprotocol Transfer
  (input [this])
  (output [this])
  (ones [this]))

(deftype FullyConnectedTraining [w b a-1 z a ones-vctr activ-fn]
  Releaseable
  (release [_]
    (release w)
    (release b)
    (release a-1)
    (release z)
    (release a)
    (release ones))
  Parameters
  (weights [_] w)
  (bias [_] b)
  Transfer
  (input [_] a-1)
  (output [_] a)
  (ones [_] ones-vctr)
  Backprop
  (forward [_]
    (activ-fn (rk! -1.0 b ones-vctr (mm! 1.0 w a-1 0.0 z)) a))
  (backward [_]
    (throw (ex-info "TODO"))))

(defn sigmoid!
  ([x]
   (linear-frac! 0.5 (tanh! (scal! 0.5 x)) 0.5))
  ([x y]
   (linear-frac! 0.5 (tanh! (scal! 0.5 (copy! x y))) 0.5)))

(defn training-layer
  ([inference-layer input ones-vctr]
   (let-release [w (view (weights inference-layer))
                 b (view (bias inference-layer))
                 a-1 (view input)
                 z (ge w (mrows w) (dim ones-vctr))
                 a (raw z)
                 o (view ones-vctr)]
     (->FullyConnectedTraining w b a-1 z a o (activation-fn inference-layer))))
  ([inference-layer previous-backprop]
   (training-layer inference-layer
                   (output previous-backprop)
                   (ones previous-backprop))))

#_(with-release [x (ge native-float 2 2 [0.3 0.9 0.3 0.9])
               ones (vctr native-float 1 1)
               layer-1 (fully-connected native-float tanh! 2 4)
               layer-2 (fully-connected native-float sigmoid! 4 1)
               training-layer-1 (training-layer layer-1 x ones)
               training-layer-2 (training-layer layer-2 training-layer-1)]
  (transfer! [0.3 0.1 0.9 0.0 0.6 2.0 3.7 1.0] (weights layer-1))
  (transfer! [0.7 0.2 1.1 2] (bias layer-1))
  (transfer! [0.75 0.15 0.22 0.33] (weights layer-2))
  (transfer! [0.3] (bias layer-2))
  (forward training-layer-1)
  (forward training-layer-2)
  (transfer (output training-layer-2)))


#_
(with-platform (first (platforms))
  (let [dev (second (devices))
        ctx (context [dev])
        q   (command-queue-1 ctx dev)]
    (with-context ctx
      (with-queue q
        (with-default-engine
        (with-release [factory (opencl-float ctx q) ]


          )   
          )))))
(declare ^:dynamic the-var)

(defmacro my-example
  [& body]
  `(let [the-var# "test" ]
      ~@body) )
#_
(my-example (println the-var))
#_
(with-open)

(declare ^:dynamic *factory*)
#_
(defmacro with-open-cl
  "  "
  [& body]
  `(with-platform (first (platforms))
    (let [dev# (second (devices))
          ctx# (context [dev#])
          q#   (command-queue-1 ctx# dev#)
          ]
      (with-context ctx#
        (with-queue q#
          (with-default-engine
            (binding [*factory* (opencl-float ctx# q#) ]
              (let [results# ~@body]
                (release *factory*)
                results#)
            )   
            ))))) )

(with-open-cl
  (with-release [x (ge *factory* 1000 1000)
                 ones (entry! (vctr *factory* 1000) 1)
                 layer-1 (fully-connected *factory* tanh! 1000 500)
                 layer-2 (fully-connected *factory* sigmoid! 500 100)
                 layer-3 (fully-connected *factory* sigmoid! 100 1)
                 training-layer-1 (training-layer layer-1 x ones)
                 training-layer-2 (training-layer layer-2 training-layer-1)
                 training-layer-3 (training-layer layer-3 training-layer-2)]
   (forward training-layer-1)
   (finish!)
   (time
     (do
       (forward training-layer-1)
       (forward training-layer-2)
       (forward training-layer-3)
       (finish!)))))


#_(with-platform (first (platforms))
  (let [dev (second (devices))
        ctx (context [dev])
        q   (command-queue-1 ctx dev)]
    (with-context ctx
      (with-queue q
        (with-default-engine
        (with-release [factory (opencl-float ctx q) ]
          (with-release [x (ge factory 1000 1000)
                   ones (entry! (vctr factory 1000) 1)
                   layer-1 (fully-connected factory tanh! 1000 500)
                   layer-2 (fully-connected factory sigmoid! 500 100)
                   layer-3 (fully-connected factory sigmoid! 100 1)
                   training-layer-1 (training-layer layer-1 x ones)
                   training-layer-2 (training-layer layer-2 training-layer-1)
                   training-layer-3 (training-layer layer-3 training-layer-2)]
            (forward training-layer-1)
            (finish!)
            (time
             (do
               (forward training-layer-1)
               (forward training-layer-2)
               (forward training-layer-3)
               (finish!))))

                )   
                )))))


#_
(uncomplicate.clojurecl.core/with-platform (clojure.core/first (uncomplicate.clojurecl.core/platforms)) 
  (clojure.core/let [neandernet.tutorial-8/dev (clojure.core/second (uncomplicate.clojurecl.core/devices)) 
                     neandernet.tutorial-8/ctx (uncomplicate.clojurecl.core/context [neandernet.tutorial-8/dev]) 
                     neandernet.tutorial-8/q (uncomplicate.clojurecl.core/command-queue-1 neandernet.tutorial-8/ctx neandernet.tutorial-8/dev)] 
    (uncomplicate.clojurecl.core/with-context neandernet.tutorial-8/ctx 
      (uncomplicate.clojurecl.core/with-queue neandernet.tutorial-8/q 
        (uncomplicate.neanderthal.opencl/with-default-engine 
          (uncomplicate.commons.core/with-release [neandernet.tutorial-8/factory (uncomplicate.neanderthal.opencl/opencl-float neandernet.tutorial-8/ctx neandernet.tutorial-8/q)] 
            (with-release [x (ge factory 1000 1000) 
                           ones (entry! (vctr factory 1000) 1) 
                           layer-1 (fully-connected factory tanh! 1000 500) 
                           layer-2 (fully-connected factory sigmoid! 500 100) 
                           layer-3 (fully-connected factory sigmoid! 100 1) 
                           training-layer-1 (training-layer layer-1 x ones) 
                           training-layer-2 (training-layer layer-2 training-layer-1) 
                           training-layer-3 (training-layer layer-3 training-layer-2)] 
              (forward training-layer-1) 
              (finish!) (time (do (forward training-layer-1) (forward training-layer-2) (forward training-layer-3) (finish!))))))))))


