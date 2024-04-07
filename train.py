import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import train_loader, test_loader
from src.model import model
from icecream import ic


# Функция для сохранения модели
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')


# Гиперпараметры
learning_rate = 0.0005
num_epochs = 100

# Loss и optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Цикл обучения
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
        data = data.view(-1, 63).float()
        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item():.3}')

# Тестирование модели
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:
        data = data.view(-1, 63).float()
        outputs = model(data, with_softmax=True)  # Получение вероятностей
        for i, out in enumerate(outputs):
            result = int((out == max(out)).nonzero(as_tuple=True)[0])
            #print(f'{labels[i]} == {result}')
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test gestures: {100 * correct / total:.3} %')


# Сохранение модели после обучения
save_model_path = './gesture_model.pth'
save_model(model, save_model_path)