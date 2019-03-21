(ns neandernet.play
  (:require [clojure.string :refer [join]])
  (:require [uncomplicate.commons.core :refer [with-release  let-release Releaseable release]]
            [uncomplicate.clojurecl
             [core :refer [sort-by-cl-version set-context! set-queue! with-platform platforms with-context context with-queue
                           sort-by-cl-version devices with-default-1 command-queue-1
                           *context* *command-queue* *platform* finish!
                           set-default-1! release-context!]]]
            [uncomplicate.neanderthal
             [core :refer [asum axpy! scal! transfer! transfer mm! rk! view-ge vctr ge entry! copy]]
             [native :refer [native-double native-float dv dge]]  
             [vect-math :refer [tanh! linear-frac! mul!]]
             [opencl :refer [opencl-float clv with-default-engine set-engine!]]]))  

(defn sigmoid! [x]
  (linear-frac! 0.5 (tanh! (linear-frac! 0.5 x 0.0)) 0.5))

(defn sigmoid-prim!
  ([x!]
   (let-release [x-copy (copy x!)]
     (sigmoid-prim! x! x-copy)))
  ([x! prim!]
   (let [x (sigmoid! x!)]
     (mul! (linear-frac! -1.0 x 1.0 prim!) x))))

(with-release [x (dv 0.1 0.5 0.9 (/ Math/PI 2.0))] [(transfer x) (sigmoid-prim! x)])

