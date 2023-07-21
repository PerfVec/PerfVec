import argparse
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor


def train(train_size, targets, epochs, h, ne):
  nums = torch.arange(1, 7).float()
  l1_sizes = nums.view(6, 1).repeat(1, 6).view(36, 1)
  l2_sizes = nums.repeat(6, 1).view(36, 1)
  dataset = torch.cat((l1_sizes, l2_sizes, targets), 1)
  test_size = 36 - train_size
  #print(dataset)

  # Construct training and testing sets.
  torch.manual_seed(1)
  train_set = torch.zeros(train_size, 3)
  test_set = torch.zeros(test_size, 3)
  sels = torch.randperm(36)
  for i in range(36):
    if i < train_size:
      train_set[i] = dataset[sels[i]]
    else:
      test_set[i - train_size] = dataset[sels[i]]
  train_set_x = train_set[:, 0:2].detach().numpy()
  train_set_y = train_set[:, 2].detach().numpy()
  test_set_x = test_set[:, 0:2].detach().numpy()
  test_set_y = test_set[:, 2].detach().numpy()
  dataset_x = dataset[:, 0:2].detach().numpy()
  dataset_y = dataset[:, 2].detach().numpy()
  #print(train_set)
  #print(test_set)
  #print(train_set_x)
  #print(train_set_y)
  #exit()

  #nn_model = MLPRegressor(hidden_layer_sizes=(h), solver='lbfgs', alpha=0, random_state=0, max_iter=epochs)
  #model = AdaBoostRegressor(nn_model, random_state=1, n_estimators=ne)
  model = MLPRegressor(hidden_layer_sizes=(h), random_state=0, alpha=0, solver='lbfgs', max_iter=epochs)
  #model = AdaBoostRegressor(random_state=1, n_estimators=ne)

  print("Training...")
  model.fit(train_set_x, train_set_y)

  print("Testing...")
  print("Train loss:", model.score(train_set_x, train_set_y))
  print("Test loss:", model.score(test_set_x, test_set_y))
  output = torch.from_numpy(model.predict(dataset_x))
  print(output)
  errors = (output - dataset[:, 2]) / dataset[:, 2]
  print("All errors:", errors)
  print("All mean error:", torch.mean(torch.abs(errors), dim=0))

  true = dataset[:, 2] * 100 * (1000 + torch.pow(2, dataset[:, 0] + 1) * 10 + torch.pow(2, dataset[:, 1] + 7)) / 1000
  print("True cost:", true)
  output = output.view(36)
  pred = output * 100 * (1000 + torch.pow(2, dataset[:, 0] + 1) * 10 + torch.pow(2, dataset[:, 1] + 7)) / 1000
  print("Predicted cost:", pred)
  pred_min = torch.min(pred)
  idx = pred == pred_min
  idx = idx.nonzero()[0, 0].item()
  print("Index", idx, "has the minimal cost", pred_min, "and cache sizes are", dataset[idx, :2])
  better_indices = true < true[idx]
  rank = better_indices.count_nonzero().item()
  print("Rank at", rank)
  return rank


def train_all(train_size, targets, epochs, h, ne):
  print("\nTrain size", train_size)
  total = 0
  for i in range(targets.shape[0]):
    print("\nBenchmark", i)
    rank = train(train_size, targets[i].view(36, 1), epochs, h, ne)
    total += rank / 36
  print("Better designs are", total / targets.shape[0])


parser = argparse.ArgumentParser(description='ASPLOS06 testing')
parser.add_argument('--train-size', type=int, default=18, metavar='N')
parser.add_argument('--hidden-size', type=int, default=2, metavar='N')
parser.add_argument('--nestimators', type=int, default=50, metavar='N')
args = parser.parse_args()
train_size = args.train_size
h = args.hidden_size
epochs = 500
targets = torch.tensor([
[96479263500,92765423500,90655460500,90616531500,89207294500,88927899500,87023587000,83855455000,81784422000,81779597000,80537983000,80243553500,81620539500,79082356500,77096966500,77108418500,76041758500,75748102500,78703272500,76663382500,74755851500,74765603500,73829336500,73531548500,77266751500,75444864500,73770909500,73782635500,72879478500,72633476500,75061271000,73524543000,72515302000,72512384000,72006763000,71907116000],
[72104533000,72096748000,72096748000,72096748000,72096748000,72096748000,54855232000,54847446000,54847446000,54847446000,54847446000,54847446000,53258151000,53250366000,53250366000,53250366000,53250366000,53250366000,36722311000,36722311000,36722311000,36722311000,36722311000,36722311000,36418965000,36418965000,36418965000,36418965000,36418965000,36418965000,36404184000,36404184000,36404184000,36404184000,36404184000,36404184000],
[2.96063E+11,3.11628E+11,3.11626E+11,3.11472E+11,3.11385E+11,3.11254E+11,2.93592E+11,3.0566E+11,3.05638E+11,3.05465E+11,3.05287E+11,3.05162E+11,2.93857E+11,2.9947E+11,2.99495E+11,2.9944E+11,2.99156E+11,2.98973E+11,2.94309E+11,2.9933E+11,2.99306E+11,2.99283E+11,2.99051E+11,2.98723E+11,2.94999E+11,2.98372E+11,2.98328E+11,2.98232E+11,2.981E+11,2.97827E+11,2.97016E+11,2.96613E+11,2.96558E+11,2.96487E+11,2.96379E+11,2.96176E+11],
[83389606000,82392035000,81632812000,81379407000,81276382000,81312403000,81240651000,80305367000,79582470000,79337183000,79218653000,79257792000,79206466000,78394871000,77622838000,77389296000,77238100000,77284059000,74455537000,73803622000,73021314000,72761163000,72613232000,72635245000,65725277000,65139914000,64910359000,64321312000,64151300000,64174154000,60501466000,60088593000,59901351000,59884324000,59303587000,59309602000],
[42586799500,42403077500,42403077500,42403077500,42403077500,42403077500,32982526000,32898738000,32898738000,32898738000,32898738000,32898738000,31577085000,31575493000,31575493000,31575493000,31575493000,31575493000,30842185500,30841266500,30841266500,30841266500,30841266500,30841266500,30797351000,30796401000,30796401000,30796401000,30796401000,30796401000,30788984500,30788984500,30788984500,30788984500,30788984500,30788984500],
[12932850500,12289888500,11569320500,10864087500,10611755000,10609305000,11125742000,10599535000,10057230000,9481868000,9270162000,9268244000,9783720000,9386852000,9038064000,8548554000,8371611000,8370404500,9204315000,8878101000,8616723000,8153970000,7995420000,7992468000,8882587000,8575919000,8384125000,7934251500,7789377000,7787845000,8677028000,8392807000,8245216000,7812081000,7700189000,7698533000],
[51535704500,51571260500,51484526500,51402551500,51268703500,51216360500,42060072500,42101547500,42034137500,41955418500,41851569500,41807459500,37707617000,37729392000,37682677000,37631344000,37584629000,37565889000,37470881000,37494925000,37467008000,37428079000,37389150000,37372827000,37244242500,37269078500,37217109500,37160839500,37075181500,37050313500,36925022500,36953141500,36834568500,36727352500,36585407500,36540491500],
[1.15559E+11,97750650500,87104644500,84214492500,82415963500,82415963500,99987050500,84269014500,76659886500,74089208500,72502581500,72502581500,79988167000,68656621000,63016839000,61151302000,59901946000,59901946000,61655513500,54961178500,51272394500,50347843000,49857335000,49857335000,51884402500,47998539500,45813398500,45037282000,44951210000,44951210000,47690346500,45358549500,43938444500,43462748500,43415648500,43415648500],
[84790218000,76408127000,71823470000,71168087000,70017607500,69936609500,74947112500,68239821500,65485102500,64861615000,63899203000,63837597500,66233192500,62820258500,61731071500,61123528500,60293160500,60260931500,62111956500,60222783500,59456528500,58873865500,58162326500,58138234500,60992199500,59621176500,59146853500,58508388500,57819248500,57785249000,60418154500,59034001500,58714188000,58172947500,57476976000,57411543500],
[71939747000,69672556000,61602651500,59834379500,58674530500,58674530500,71759917000,69491143000,61394879500,59618055500,58460882000,58460882000,71463407000,69235995000,61130111000,59327427000,58181134000,58181134000,69830432000,67733435000,60101779000,58499562000,57373722000,57373722000,65143789000,63370417000,59031572000,57757336000,56641594000,56641594000,60226013000,59530184000,58201602000,57221460000,56225941000,56225941000],
[1.34487E+11,1.34448E+11,1.34448E+11,1.34448E+11,1.34448E+11,1.34448E+11,1.15888E+11,1.15857E+11,1.15857E+11,1.15857E+11,1.15857E+11,1.15857E+11,1.04483E+11,1.04475E+11,1.04475E+11,1.04475E+11,1.04475E+11,1.04475E+11,94999926500,94999926500,94999926500,94999926500,94999926500,94999926500,93379843500,93372057500,93372057500,93372057500,93372057500,93372057500,92843558500,92835772500,92835772500,92835772500,92835772500,92835772500],
[73512847000,69642019000,69642019000,69642019000,69642019000,69642019000,56033249000,52161593000,52161593000,52161593000,52161593000,52161593000,46012678500,45209971500,45209971500,45209971500,45209971500,45209971500,37892360500,37880300500,37880300500,37880300500,37880300500,37880300500,37816868000,37814058000,37814058000,37814058000,37814058000,37814058000,37775231000,37775231000,37775231000,37775231000,37775231000,37775231000],
[72237779500,72189964500,71114099500,68396934500,64408778500,64206332500,69905614500,69815730500,68836922500,66590210500,63633866500,63472181500,68493241500,68389005500,67471689500,65450750500,63370499500,63190590500,67464435500,67367921500,66518208500,64615363500,63132058500,63019686500,66580023500,66464248500,65649266500,63854576500,62948659500,62870800500,66532296500,66383278500,65450071500,63915034500,62922570500,62844712500],
[3.28174E+11,3.45708E+11,3.45682E+11,3.45625E+11,3.4548E+11,3.45218E+11,3.28467E+11,3.43132E+11,3.43108E+11,3.43038E+11,3.42906E+11,3.42649E+11,3.28499E+11,3.3593E+11,3.359E+11,3.35852E+11,3.35753E+11,3.35553E+11,3.29132E+11,3.34589E+11,3.34562E+11,3.34515E+11,3.34422E+11,3.34235E+11,3.30311E+11,3.34582E+11,3.34558E+11,3.34512E+11,3.34414E+11,3.34223E+11,3.32444E+11,3.31728E+11,3.31705E+11,3.31668E+11,3.3158E+11,3.31411E+11],
[41513468000,41513468000,41513468000,41513468000,41513468000,41513468000,36718871000,36718871000,36718871000,36718871000,36718871000,36718871000,33899192000,33899192000,33899192000,33899192000,33899192000,33899192000,32514002500,32514002500,32514002500,32514002500,32514002500,32514002500,32512162500,32512162500,32512162500,32512162500,32512162500,32512162500,32511976000,32511976000,32511976000,32511976000,32511976000,32511976000],
[79540460500,80065755500,80125704500,79782611500,79696967500,79658038500,64219268500,64120949500,64076203500,63912341500,63919247500,63898035500,61923768500,62200817000,62150967500,61986730500,61967900500,61960053500,61621197000,61530594000,61482576000,61324951000,61324951000,61317165000,61237895500,60822655500,60758786500,60857129000,60633888000,60610530000,60649762000,60628447000,60762530000,60611074000,60455343000,60431986000],
[34078743500,34078743500,34078743500,34078743500,34078743500,34078743500,19144200500,19144200500,19144200500,19144200500,19144200500,19144200500,17602188500,17602188500,17602188500,17602188500,17602188500,17602188500,17414608500,17414608500,17414608500,17414608500,17414608500,17414608500,17407541500,17407541500,17407541500,17407541500,17407541500,17407541500,17398777500,17398777500,17398777500,17398777500,17398777500,17398777500]
], dtype=torch.float)
targets /= 1000000000000
#print(targets)
print(targets.shape)

if train_size == 0:
  for train_size in range(1, 36):
    train_all(train_size, targets, epochs, h, args.nestimators)
else:
  train_all(train_size, targets, epochs, h, args.nestimators)
