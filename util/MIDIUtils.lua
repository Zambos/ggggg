local MIDI = require 'MIDI'
local MIDIUtils = {}
MIDIUtils.__index = MIDIUtils
--local midifile = assert(io.open('util/f1.mid','rb'))
--local myfile = MIDI.midi2score(midifile:read("*all"))
--print(myfile)
--local data = {}
--local n=294
--for i=1, n do
--  if myfile[3][i][1]=="note" then
--    table.insert(data,myfile[3][i][5])
--  end
--end
--print(data)
--
--midifile:close()

--opus
-----------------------------------------
-- Time sinature v [2] - muzikalnoto vreme primer (4/4)
--[2] - ?
--[3] - 4
--[4] - 2 - denum stepen na 2
--[5] - 24 - clocks per tick
--[6] - 8 - broi 1/8 za 24 tika
--
--
--Key Signature v [2] -
--?
--
--Set Tempo v [2] -
--120 bpm = 500000
--1bpm = 4166,666666667
--rawdata[1] - pulses per quater note
--
--
--scor
--{"note", start_time, duration, channel, note, velocity}
--=====================================TEST==ZONE==========================================================




--===================================================================================================================
--function MIDIUtils.midi_to_tensor(in_textfile, out_vocabfile, out_tensorfile)
--    local timer = torch.Timer()
--
--    print('loading text file...')
--    local cache_len = 10000
--    local rawdata
--    local tot_len = 0
--    local f = assert(io.open(in_textfile, "r"))
--
--    print('putting data into tensor...')
--    local data = torch.ByteTensor(tot_len) -- store it into 1D first, then rearrange
--    f = assert(io.open(in_textfile, "r"))
--    local currlen = 0
--    rawdata = f:read(cache_len)
--    repeat
--        for i=1, #rawdata do
--            data[currlen+i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
--        end
--        currlen = currlen + #rawdata
--        rawdata = f:read(cache_len)
--        --print(rawdata)
--    until not rawdata
--    f:close()
--    print("======================================")
--    print(data)
--
--    -- save output preprocessed files
--    --print('saving ' .. out_vocabfile)
--    --torch.save(out_vocabfile, vocab_mapping)
--    print('saving ' .. out_tensorfile)
--    torch.save(out_tensorfile, data)
--    --print(vocab_mapping)
--    --print(data)
--end
--


-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

function MIDIUtils.create(data_dir, batch_size, seq_length, split_fractions)
  -- split_fractions is e.g. {0.9, 0.05, 0.05}

  local self = {}
  setmetatable(self, MIDIUtils)

  local input_file = path.join(data_dir, 'f1.mid')
  --Vocab file sadarja vsicki srshtani noti s vsicki vrmna

  local vocab_file = path.join(data_dir, 'vocab.t7')
  local tensor_file = path.join(data_dir, 'data.t7')

  -- fetch file attributes to determine if we need to rerun preprocessing
  local run_prepro = true
  --if not (path.exists(vocab_file) or path.exists(tensor_file)) then
  --  -- prepro files do not exist, generate them
  --print('vocab.t7 and data.t7 do not exist. Running preprocessing...')
  --run_prepro = true
  --else

  -------------------
  -- check if the input file was modified since last time we
  -- ran the prepro. if so, we have to rerun the preprocessing
  local input_attr = lfs.attributes(input_file)
  local vocab_attr = lfs.attributes(vocab_file)
  local tensor_attr = lfs.attributes(tensor_file)
  --if input_attr.modification > vocab_attr.modification or input_attr.modification > tensor_attr.modification then
  ---print('vocab.t7 or data.t7 detected as stale. Re-running preprocessing...')
  --run_prepro = true
  -- end
  --end
  if true--run_prepro then
  then -- construct a tensor with all the data, and vocab file
    print('one-time setup: preprocessing input text file ' .. input_file .. '...')
    --------------------------------------------------------------------------
    --------------------------------------------------------------------------
    --------------------------------------------------------------------------
    MIDIUtils.text_to_tensor(input_file, vocab_file, tensor_file)
  end

  print('loading data files...')
  local data = torch.load(tensor_file)
  self.vocab_mapping = torch.load(vocab_file)

  -- cut off the end so that it divides evenly
  local len = data:size(1)
  --batch_size = 2
  --seq_length = len/3
  if len % (batch_size * seq_length) ~= 0 then
    print('cutting off end of data so that the batches/sequences divide evenly')
    print(len.." "..batch_size * seq_length
      * math.floor(len / (batch_size * seq_length)))
    data = data:sub(1, batch_size * seq_length
      * math.floor(len / (batch_size * seq_length)))
  end
  print(#data)

  -- count vocab
  self.vocab_size = 0
  for _ in pairs(self.vocab_mapping) do
    self.vocab_size = self.vocab_size + 1
  end
  self.vocab_size = self.vocab_size*6

  -- self.batches is a table of tensors
  print('reshaping tensor...')
  self.batch_size = batch_size
  self.seq_length = seq_length

  local ydata = data:clone()
  ydata:sub(1,-2):copy(data:sub(2,-1))
  ydata[-1] = data[1]
  --ydata is data backwards
  --Data is one dimentioal
  --print(data:dim())
  self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
  --print(data:view(batch_size,-1))
  --self.x_batches = self.split(self.x_batches, data, seq_length, 1)
  --print(data:view(batch_size,-1))
  --print(self.x_batches[1])
  --os.exit()
  self.nbatches = #self.x_batches
  self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
  --self.y_batches = self.split(self.y_batches,ydata, seq_length, 1)
  assert(#self.x_batches == #self.y_batches)

  -- lets try to be he lpful here
  if self.nbatches < 50 then
    print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
  end

  -- perform safety checks on split_fractions
  assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
  assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
  assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
  if split_fractions[3] == 0 then
    -- catch a common special case where the user might not want a test set
    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = self.nbatches - self.ntrain
    self.ntest = 0
  else
    -- divide data to train/val and allocate rest to test
    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = math.floor(self.nbatches * split_fractions[2])
    self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
  end

  self.split_sizes = {self.ntrain, self.nval, self.ntest}
  self.batch_ix = {0,0,0}

  print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
  collectgarbage()
  return self
end

function MIDIUtils:reset_batch_pointer(split_index, batch_index)
  batch_index = batch_index or 0
  self.batch_ix[split_index] = batch_index
end

function MIDIUtils:next_batch(split_index)
  --print(self.batch_ix[split_index])
  --os.exit()
  if self.split_sizes[split_index] == 0 then
    -- perform a check here to make sure the user isn't screwing something up
    local split_names = {'train', 'val', 'test'}
    print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
    os.exit() -- crash violently
  end
  -- split_index is integer: 1 = train, 2 = val, 3 = test
  self.batch_ix[split_index] = self.batch_ix[split_index] + 1
  if self.batch_ix[split_index] > self.split_sizes[split_index] then
    self.batch_ix[split_index] = 1 -- cycle around to beginning
  end
  -- pull out the correct next batch
  local ix = self.batch_ix[split_index]
  if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
  if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
  --print(self.x_batches[ix])
  --os.exit()
  return self.x_batches[ix], self.y_batches[ix]
end

-- *** STATIC method *** RABOI PRAVLNO
function MIDIUtils.text_to_tensor(in_midifile, out_vocabfile, out_tensorfile)
  local timer = torch.Timer()
  local vocab_mapping2 = {}
  local vocab_mapping = {}


  dir='data/midi/'
  datas = {}
  i=0
  p = io.popen('ls '..dir..'*.mid ')  --Open directory look for files, save data in p. (with option "/b" everything contained in the given directory is listed with simple format)
  for file in p:lines() do                    --Loop through all files
    if file == "" then break end
    datas[i] = tostring(file)
    i = i + 1
  end

  print(datas)
  local tot_len = 0
  local indexer = 1
  local temp_len = 0
  local unordered = {}
  for i=0,#datas do
    --if f == "" then break end
    print('loading midi file...')
    local rawdata
    local f = assert(io.open(datas[i], "r"))
    local rawdata = MIDI.midi2score(f:read("*all"))

    if rawdata[3]~=nil then indexer = 3 elseif rawdata[2]~=nil then indexer=2 end
    --for n, f in in_midifile do
    --print(rawdata[indexer])
    -- create vocabulary if it doesn't exist yet
    print('creating vocabulary mapping...')
    -- record all characters to a set
    --print(rawdata[indexer][#rawdata[indexer]][2])
    
    for i=1, #rawdata[indexer] do
      if rawdata[indexer][i][1]=="note" then
        tot_len= tot_len+1
        if not unordered[rawdata[indexer][i][5]] then unordered[rawdata[indexer][i][5]] = true end
      end
    end
    --print(unordered)
    --print(tot_len)
    f:close()
  end
  --print(unordered)

  -----------------------------------------------------------------------------------
  -- unordered e napalnen
  -- sort into a table (i.e. keys become 1..N)

  local ordered = {}

  for char in pairs(unordered) do ordered[#ordered + 1] = char end
  table.sort(ordered)
  --print(ordered)
  -- invert `ordered` to create the char->int mapping
  temp_len = temp_len+ #ordered
  local temp = 0;
  for i, char in ipairs(ordered) do
    temp= temp+1
    --print(i)
    --os.exit()
    vocab_mapping2[char]={1+i+temp_len*0,1+i+temp_len*1,1+ i+temp_len*2, 1+i+temp_len*3, 1+i+temp_len*4, 1+i+temp_len*5}
  end
  vocab_mapping2[0]={1+temp_len*0,1+temp_len*1, 1+temp_len*2, 1+temp_len*3, 1+temp_len*4, 1+temp_len*5}
  --print(vocab_mapping2)
  --vocab_mapping[-1]=temp+1
  for c,i in pairs(vocab_mapping2) do

    vocab_mapping[127*0 + c] = i[1]
    vocab_mapping[127*1 + c] = i[2]
    vocab_mapping[127*2 + c] = i[3]
    vocab_mapping[127*3 + c] = i[4]
    vocab_mapping[127*4 + c] = i[5]
    vocab_mapping[127*5 + c] = i[6]
    --map[temp]=c

    --io.write(ivocab[i],"/n")
  end
  --print(vocab_mapping)
  print('putting data into tensor...')
  --tot_len=math.floor(rawdata[3][#rawdata[indexer]][2]/(rawdata[1]/4))
  --print(tot_len)
  local data = torch.ByteTensor(tot_len) -- store it into 1D first, then rearrange
  data= data:fill(1)
  local currlen = 0
  for i=0,#datas do
    f = assert(io.open(datas[i], "r"))
    local currtime = 0;
    local rawdata = MIDI.midi2score(f:read("*all"))

    if rawdata[3]~=nil then indexer = 3 elseif rawdata[2]~=nil then indexer=2 end
    local currnote = 0;
    for i=1, #rawdata[indexer] do
      if rawdata[indexer][i] ~= nil then
        if rawdata[indexer][i][1]=="note" then
          currnote= currnote+1
          local tempIndex = math.abs(math.floor((math.log10(rawdata[indexer][i][3]/(rawdata[1]/4))/math.log10(2)) +0.5))
          --print(tempIndex.."  "..rawdata[indexer][i][5].." "..rawdata[1].." "..rawdata[indexer][i][3].."   "..(rawdata[1]/4).."  "..math.log10(rawdata[indexer][i][3]/(rawdata[1]/4))/math. log10(2))
          --print(rawdata[indexer][i][5]+ 127*tempIndex.."   "..vocab_mapping[rawdata[indexer][i][5]+ 127*tempIndex])
          --data[1]=3
         -- print(math.floor(currlen+(rawdata[indexer][i][2] )/(rawdata[1]/4)+1).." "..rawdata[indexer][i][2].."  "..(currlen+(rawdata[indexer][i][2] )/(rawdata[1]/4)+1).." "..(vocab_mapping[rawdata[indexer][i][5]+ 127*tempIndex]))
          --print(currlen+currnote)
          
          data[currlen+currnote] = vocab_mapping[rawdata[indexer][i][5]+ 127*tempIndex]
          --print(data[currlen+currnote].."   "..vocab_mapping[rawdata[indexer][i][5]+ 127*tempIndex])
         if data[currlen+currnote]==0 then data[currlen+currnote]=data[currlen+currnote]+1
         --print(data[currlen+currnote].."   "..vocab_mapping[rawdata[indexer][i][5]+ 127*tempIndex]) 
         --os.exit()
         end
        end 
      end
    end
    currlen = currlen + currnote 
    f:close()
  end
  --print(rawdata)
  --print(rawdata[indexer])
  --data:narrow(1,currlen)
  print("======================================")

  --print(data)
  --os.exit()
  -- save output preprocessed files
  print('saving ' .. out_vocabfile)
  torch.save(out_vocabfile, vocab_mapping)
  print('saving ' .. out_tensorfile)
  torch.save(out_tensorfile, data)
  --print(vocab_mapping)
end
return MIDIUtils

