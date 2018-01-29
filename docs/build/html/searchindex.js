Search.setIndex({docnames:["dataset","evaluator","index","loss","models","notes/intro","optim","trainer","util"],envversion:52,filenames:["dataset.rst","evaluator.rst","index.rst","loss.rst","models.rst","notes/intro.md","optim.rst","trainer.rst","util.rst"],objects:{"seq2seq.dataset":{fields:[0,0,0,"-"]},"seq2seq.dataset.fields":{SourceField:[0,1,1,""],TargetField:[0,1,1,""]},"seq2seq.dataset.fields.TargetField":{SYM_EOS:[0,2,1,""],SYM_SOS:[0,2,1,""],build_vocab:[0,3,1,""]},"seq2seq.evaluator":{evaluator:[1,0,0,"-"],predictor:[1,0,0,"-"]},"seq2seq.evaluator.evaluator":{Evaluator:[1,1,1,""]},"seq2seq.evaluator.evaluator.Evaluator":{evaluate:[1,3,1,""]},"seq2seq.evaluator.predictor":{Predictor:[1,1,1,""]},"seq2seq.evaluator.predictor.Predictor":{predict:[1,3,1,""]},"seq2seq.loss":{loss:[3,0,0,"-"]},"seq2seq.loss.loss":{Loss:[3,1,1,""],NLLLoss:[3,1,1,""],Perplexity:[3,1,1,""]},"seq2seq.loss.loss.Loss":{eval_batch:[3,3,1,""],get_loss:[3,3,1,""],reset:[3,3,1,""]},"seq2seq.models":{DecoderRNN:[4,0,0,"-"],EncoderRNN:[4,0,0,"-"],TopKDecoder:[4,0,0,"-"],attention:[4,0,0,"-"],baseRNN:[4,0,0,"-"],seq2seq:[4,0,0,"-"]},"seq2seq.models.DecoderRNN":{DecoderRNN:[4,1,1,""]},"seq2seq.models.EncoderRNN":{EncoderRNN:[4,1,1,""]},"seq2seq.models.EncoderRNN.EncoderRNN":{forward:[4,3,1,""]},"seq2seq.models.TopKDecoder":{TopKDecoder:[4,1,1,""]},"seq2seq.models.TopKDecoder.TopKDecoder":{forward:[4,3,1,""]},"seq2seq.models.attention":{Attention:[4,1,1,""]},"seq2seq.models.attention.Attention":{set_mask:[4,3,1,""]},"seq2seq.models.baseRNN":{BaseRNN:[4,1,1,""]},"seq2seq.models.seq2seq":{Seq2seq:[4,1,1,""]},"seq2seq.optim":{optim:[6,0,0,"-"]},"seq2seq.optim.optim":{Optimizer:[6,1,1,""]},"seq2seq.optim.optim.Optimizer":{set_scheduler:[6,3,1,""],step:[6,3,1,""],update:[6,3,1,""]},"seq2seq.trainer":{supervised_trainer:[7,0,0,"-"]},"seq2seq.trainer.supervised_trainer":{SupervisedTrainer:[7,1,1,""]},"seq2seq.trainer.supervised_trainer.SupervisedTrainer":{train:[7,3,1,""]},"seq2seq.util":{checkpoint:[8,0,0,"-"]},"seq2seq.util.checkpoint":{Checkpoint:[8,1,1,""]},"seq2seq.util.checkpoint.Checkpoint":{INPUT_VOCAB_FILE:[8,2,1,""],MODEL_NAME:[8,2,1,""],OUTPUT_VOCAB_FILE:[8,2,1,""],TRAINER_STATE_NAME:[8,2,1,""],load:[8,4,1,""],path:[8,2,1,""],save:[8,3,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","classmethod","Python class method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:classmethod"},terms:{"case":7,"class":[0,1,3,4,6,7,8],"default":[1,4,6,7,8],"float":[1,3,4,6,7],"function":[3,4,5,6],"import":[],"int":[1,3,4,6,7,8],"new":5,"return":[1,3,4,7,8],"true":[0,3,4,7],"try":5,"while":[],Adding:[],EOS:5,For:[0,3,5],IDs:4,The:[3,4,5,6,7,8],There:5,Use:5,Used:4,_loss:3,about:0,acc_loss:3,accumul:3,accuraci:[1,5],activ:[],adam:7,addit:[4,5],after:[3,5,7],against:1,all:5,allow:[4,8],alpha:[],alreadi:[],altern:8,ani:5,append:0,appli:4,applic:[],appreci:[],arbitrari:4,architectur:4,arg:[0,4,8],argument:4,attend:4,attent:5,attn:4,avail:5,averag:3,base:[3,4],batch:[1,3,4,5,7],batch_first:0,batch_siz:[1,4,7],beam:4,becom:4,befor:5,being:8,below:5,benchmark:[],bidirect:4,bidirection:5,bool:[3,4,7],both:5,bug:[],build_vocab:0,calcul:3,call:[3,8],caller:6,can:[3,5,8],caption:[],cell:4,chang:5,checkout:[],checkpoint:7,checkpoint_everi:7,checkpoint_path:[5,7],classmethod:8,clip:6,cluster:8,cnn:[],coco:[],collabor:[],com:[0,5],command:5,commandlin:5,commit:5,commonli:3,complet:5,compon:[],conda:[],configur:4,constantli:[],contain:[4,5],context:4,contribut:2,convers:[],convolut:[],copi:8,correspond:[],could:6,cpu:[],creat:[],criteria:6,criterion:3,ctrl:5,current:[5,6,7,8],data:[0,1,4,5,7,8],dataset:[1,2,5,7],decod:4,decode_funct:4,decoder_hidden:4,decoder_output:4,decoder_rnn:4,defin:3,defulat:[],depend:[3,6],detail:[4,7],dev:[5,7],dev_data:7,dev_path:5,develop:[5,6],dictionari:4,differ:8,dim:4,dimens:4,directli:[3,4],directori:[5,7,8],disabl:6,discov:5,discuss:[],disk:8,diverg:5,doc:3,docstr:[],document:[],drawn:4,dropout:4,dropout_p:4,dure:[5,7,8],each:[3,4],embed:[4,5],embedding_s:[4,5],encapsul:[3,6],encod:4,encoder_hidden:4,encoder_output:4,encodr:[],end:[0,4],enter:5,environ:5,eos:0,eos_id:[0,4],epoch:[5,6,7,8],especi:[],etc:[],eval_batch:3,evalu:[2,3],everi:4,evolv:[],exampl:[4,8],exist:5,expect:[3,4,5],experi:[7,8],experiment_dir:8,exponenti:3,expt_dir:[5,7],extens:[],facilit:[],fals:[4,7],fast:[],featur:4,feedback:[],feel:[],field:8,file:8,fix:[],flag:4,flexibl:[],focu:[],folder:[5,7],follow:[4,8],forc:[0,4,5,7],fork:5,format:8,forward:4,forward_rnn:[],framework:[4,7],free:[],frequent:[],from:[4,5,7,8],full:8,func:4,gener:[4,5],get:3,get_loss:3,github:[0,5],give:8,given:[1,3,4,6,7],goal:[],googl:[],gradient:6,gru:[4,5],guid:[],had:[],has:5,have:[],heavi:5,help:[5,7],here:[],hidden:[4,5],hidden_s:[4,5],how:[3,7],html:3,http:[0,3,5],ibm:5,illustr:5,imag:[],implement:[3,5],improv:[],includ:[5,6],include_length:0,incom:4,index:[0,3,4],indic:4,individu:3,infer:[],inferenc:3,inform:[0,3,4],initi:[4,8],input:[1,4,5,8],input_dropout_p:4,input_len:4,input_length:4,input_s:4,input_var:4,input_vari:4,input_vocab:[4,8],input_vocab_fil:8,instal:5,instanti:6,integ:4,integr:5,integration_test:5,interfac:3,introduct:2,invers:5,issu:[],item:[],its:5,job:8,k80:[],kei:4,key_attn_scor:4,key_input:4,key_length:4,key_sequ:4,keyword:4,kind:[],kwarg:[0,4],languag:[1,3,8],last:4,later:8,latest:7,layer:[4,5],learing_r:7,learn:[5,6,7],learning_r:7,least:[],length:4,less:[],librari:5,like:[],likelihood:3,line:5,linear:4,linear_out:4,list:[1,3,4],load:[0,5,7,8],local:8,log:3,log_softmax:4,logic:3,look:4,loop:8,loss:[1,2,5,6,7],lr_schedul:6,lstm:4,machin:5,major:[],make:[1,7,8],manag:[0,8],mani:7,mask:[3,4],master:3,max_grad_norm:[6,7],max_len:4,max_length:4,max_seq_length:4,maximum:4,mechan:4,messag:3,met:6,method:[3,8],mini:4,minut:[],mode:5,model:[1,3,5,7,8],model_checkpoint:[],model_nam:8,modul:4,modular:[],more:0,multi:4,multipl:3,must:4,n_layer:4,name:[3,5,8],necessari:6,need:[],neg:3,nllloss:[1,7],none:[3,4,7,8],norm:6,norm_term:3,normal:3,note:[2,4],num_direct:4,num_epoch:7,num_lay:4,number:[4,5,6,7,8],numpi:5,object:[1,4,6,7,8],onc:5,one:[3,4],onli:[],open:[],optim:[2,5,7,8],option:[1,3,4,5,6,7,8],org:3,organ:[],origin:5,our:[],out:5,output:[3,4,5,8],output_dir:5,output_len:4,output_vocab:8,output_vocab_fil:8,overrid:3,overview:5,overwritten:7,own:3,packag:[2,6],param:[6,8],paramet:[1,3,4,6,7,8],pass:8,path:[7,8],perform:[1,6],pip:5,pleas:[0,3,5],pre:1,predict:[1,4,5],prepend:0,preprocess:0,previous:8,print:5,print_everi:7,probabl:4,problem:[],proce:[],process:[0,4],project:5,prompt:5,propos:[],provid:[4,6],publish:[],pull:5,python:5,pytorch:[0,3,5,7],qualiti:[],question:[],quickstart:2,randn:4,random:4,random_se:7,rate:[6,7],ratio:7,recommend:[],recurr:4,refer:[2,3,5],regard:3,relat:8,releas:[],report:[],repositori:5,repres:4,request:5,requir:2,reset:3,respect:5,result:3,resum:[5,7,8],ret_dict:4,retain_output_prob:4,retrain:5,revers:5,right:5,rnn:4,rnn_cell:[4,5],root:8,run:[5,7,8],same:3,sampl:4,save:8,schedul:6,scratch:5,script:[],search:4,seen:8,sentenc:[0,4],seq2seq:[0,1,3,5,6,7,8],seq_len:4,sequenc:[0,4,5],sequenti:8,set:[4,5,6,7],set_mask:4,set_schedul:6,setup:[],setuptool:[],sever:5,sgd:6,should:[6,7],shown:[],simpl:5,singl:6,size:[1,4,5,7],size_averag:3,small:[],smaller:4,sos:0,sos_id:[0,4],sourc:[0,1,3,4,5,6,7,8],sourcefield:0,specifi:4,src_seq:1,src_vocab:1,standard:4,start:[0,4],state:[4,5,8],step:[0,4,6,8],steplr:6,store:[3,5,7,8],str:[3,4,7,8],structur:[],sub:[3,4],subdirectori:8,substanti:5,supervis:[6,7],supervisedtrain:7,support:[],suspend:8,sym_eo:[0,4],sym_mask:4,sym_so:0,symbol:[0,4],system:[],take:[],target:[1,3,4,5],target_vari:4,targetfield:0,teach:7,teacher:[4,5],teacher_forcing_ratio:[4,7],techniqu:[],tensor:[3,4],term:3,termin:5,tesla:[],test:5,test_data:5,text:0,tgt_seq:1,tgt_vocab:1,than:4,them:3,thi:[3,4,5],think:5,those:8,three:5,through:8,time:8,timestamp:[],toi:5,token:[1,3,4],tool:5,toolkit:5,top:[4,5],top_k:7,topk_length:4,topk_sequ:4,torch:[3,4,6],torchtext:[0,5],torcn:3,toy_revers:[],train:[1,3,6,7,8],train_model:5,train_path:5,trainer:[2,5,6,8],trainer_st:8,trainer_state_nam:8,transform:4,translat:5,txt:[],type:[1,3,4,7,8],uniformli:4,unittest:5,updat:6,usabl:[],usag:5,use:[0,3,4,5],use_attent:4,used:[3,4,5,6,7],uses:[5,6],using:[4,5,8],util:2,vagrant:[],vagrantfil:[],valu:[3,4,6],variabl:[0,3,4,8],variable_length:4,verifi:[],version:5,virtual:[],virtualenv:[],visit:5,vocab:8,vocab_s:4,vocabulari:[4,5,8],volatil:4,websit:5,weight:[3,4],welcom:5,when:[3,4,6,8],where:4,whether:4,which:[4,5],whose:4,within:[4,8],wmt:[],would:[4,7],wrapper:0,write:8,y_m_d_h_m_:8,you:5,your:[3,5],yyyy_mm_dd_hh_mm_ss:[]},titles:["Dataset","Evaluator","Codebase for the i-machine-think project, a modular and fully tested Pytorch implementation for seq2seq models","Loss","Models","Introduction","Optim","Trainer","Util"],titleterms:{attent:4,basernn:4,build:[],checkpoint:[5,8],code:[],codebas:2,contribut:5,dataset:0,decoderrnn:4,develop:[],encoderrnn:4,environ:[],evalu:[1,5],exampl:5,field:0,framework:[],from:[],fulli:2,get:[],implement:2,infer:5,instal:[],introduct:5,librari:[],loss:3,machin:2,model:[2,4],modular:2,nllloss:3,optim:6,perplex:3,plai:[],predictor:1,prepar:[],prerequisit:[],project:2,pull:[],pytorch:2,quickstart:5,request:[],requir:5,roadmap:[],script:5,seq2seq:[2,4],sequenc:[],sourc:[],start:[],statu:[],style:[],supervised_train:7,test:2,think:2,toi:[],topkdecod:4,train:5,trainer:7,troubleshoot:[],util:8}})