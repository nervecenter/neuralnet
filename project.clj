(defproject neuralnet-clj "0.5.0"
  :description "A simple neural net in Clojure"
  ;; :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
								 [org.clojure/tools.namespace "0.2.11"]
								 [net.mikera/vectorz-clj "0.44.1"]
								 [org.clojure/math.numeric-tower "0.0.4"]]
  :main ^:skip-aot neuralnet-clj.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}}
	:repl-options {:init-ns neuralnet-clj.core
								 :init (require '[neuralnet-clj.core-test :as tests]
																'[clojure.test :refer [run-tests]])})
