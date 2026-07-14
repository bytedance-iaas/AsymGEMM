I am doing one scheduling exploration to tune between latency mode and memory mode of this system. I also tried to improved the source code to reduce latency and even expand the memory capacity even more. So there are tons of work modified here.

On another machine i hav ethe same system but i was working on expert paralleim implemnatino and better ceiling search harness etc. plotting results etc.

Currently I wanna merge them safely wihtou losing proigress or breaking any code. This is the current repo /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM and the other repo is at /home/kevinni/AsymGEMM-SFT-38/third_party/AsymGEMM
Please expore both comprehensivelt adn extenisvel i wanna merge AsymGEMM-SFT-38/third_party/AsymGEMM's changes onto us /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

Explore extensivel tand compreeilve. Isn there any conflcits? if so we need to be ultra careful and ultra metricou in decindhwo to resolve any confflits. Or si tehr any errors or incompaitl as epcts that cannot be triviallt resolved. 
Expopre and let me know what are conflicts that cannot be triviallt explored and that require  mwe to intervene. 

SUPER improtnat after mergin we need to redo some testins wiht large owklodas to ensure that eveyrhig still wokrs all right. 
Note taht we NEVENR RUN on thi hsot directl. we need to run by acting asym40_enroot_run this starts the continer wehre we can acutka run the code this is suerp improant we cNANOT ru the code on teh host directly.

We need to push all the curen hagne to liek main_kevin_scheduler so that we keep this backup right?
