# Copyright 2023 Haoru Xue
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class DummyModel(nn.Module):
    def __init__(self, n_in, n_out):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(n_in, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, n_out)
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def export_to_torch_script(self, filename):
        example_input = torch.rand(1, self.n_in)
        traced_script_module = torch.jit.trace(self, example_input)
        traced_script_module.save(filename)


if __name__ == "__main__":
    torch.random.manual_seed(0)
    model = DummyModel(8, 6)

    example_input = torch.ones(1, 8)
    example_output = model(example_input)
    print("Example output:", example_output)
    example_jacobian = torch.autograd.functional.jacobian(model, example_input)
    print("Example Jacobian:", example_jacobian)

    # model.export_to_torch_script("dummy_model.pt")

    # move the model to GPU and benchmark
    model = model.cuda()
    example_input = example_input.cuda()
    start_time = time.time()
    for i in range(1000):
        example_output = model(example_input)
    end_time = time.time()
    print("Average time for 1000 forward passes:", (end_time - start_time) / 1000, "s")
    start_time = time.time()
    for i in range(1000):
        example_jacobian = torch.autograd.functional.jacobian(model, example_input)
    end_time = time.time()
    print("Average time for 1000 Jacobian calculations:", (end_time - start_time) / 1000, "s")
