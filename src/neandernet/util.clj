(ns neandernet.util
  (:require [uncomplicate.commons.core :refer [with-release release ]]
            [uncomplicate.clojurecl.core :refer [with-context with-queue devices platforms with-platform context command-queue-1]]
            [uncomplicate.neanderthal [opencl :refer [with-default-engine opencl-float]]]))

(declare ^:dynamic *factory*)
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
#_(defmacro with-open-cl
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

#_(defmacro with-open-cl
  "  "
  [& body]
  `(with-platform (first (platforms))
    ~(let [dev (second (devices))
           ctx (context [dev])
           q   (command-queue-1 ctx dev)]
      (with-context ctx
        (with-queue q
          (with-default-engine
            (with-release [factory (opencl-float ctx q) ]
              ~@body
            )))))))
