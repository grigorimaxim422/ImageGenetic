def forward(self,
    x: Tensor) -> Tensor:
  stem = self.stem
  x0 = (stem).forward(x, )
  fire2 = self.fire2
  f2 = (fire2).forward(x0, )
  fire3 = self.fire3
  f3 = torch.add((fire3).forward(f2, ), f2)
  fire4 = self.fire4
  f4 = (fire4).forward(f3, )
  maxpool = self.maxpool
  f40 = (maxpool).forward(f4, )
  fire5 = self.fire5
  f5 = torch.add((fire5).forward(f40, ), f40)
  fire6 = self.fire6
  f6 = (fire6).forward(f5, )
  fire7 = self.fire7
  f7 = torch.add((fire7).forward(f6, ), f6)
  fire8 = self.fire8
  f8 = (fire8).forward(f7, )
  maxpool0 = self.maxpool
  f80 = (maxpool0).forward(f8, )
  fire9 = self.fire9
  f9 = (fire9).forward(f80, )
  conv10 = self.conv10
  c10 = (conv10).forward(f9, )
  avg = self.avg
  x1 = (avg).forward(c10, )
  x2 = torch.view(x1, [torch.size(x1, 0), -1])
  return x2
