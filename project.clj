(defproject neandernet "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [uncomplicate/neanderthal "0.22.0"]
                 [criterium "0.4.4"]
;                 [lein-cljfmt "0.6.4"]
                 [org.slf4j/slf4j-simple "1.7.5"]
                 ]
  :exclusions [[org.jcuda/jcuda-natives :classifier "apple-x86_64"]
               [org.jcuda/jcublas-natives :classifier "apple-x86_64"]] 
  :repl-options {:init-ns neandernet.core})
