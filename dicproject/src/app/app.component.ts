import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';

import { ReactiveFormsModule } from "@angular/forms";
import { FormControl, FormGroup } from '@angular/forms';

import { PredictService } from "./services/predict.service";

import { SafeResourceUrl} from '@angular/platform-browser';

import { HttpResponse } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, ReactiveFormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
  providers: [ PredictService ]
})
export class AppComponent {
  title = 'dicproject';
  

  // DetailsForm: FormGroup;

  DetailsForm = new FormGroup({
    Age: new FormControl(),
    Gender: new FormControl(),
    DepCount: new FormControl(),
    MonBook: new FormControl(),
    TotRelCnt: new FormControl(),
    MonInactCnt: new FormControl(),
    CredLim: new FormControl(),
    TotRevBal: new FormControl(),
    TotAmtChng: new FormControl(),
    TotTransAmt: new FormControl(),
    TotTransCnt: new FormControl(),
    TotCntChng: new FormControl(),
    AvgUtil: new FormControl(),
    EduLvl: new FormControl(),
    Marital: new FormControl(),
    IncCat: new FormControl(),
    CardCat: new FormControl(),
  });

  pred: string = "lalala"
  showRes: boolean = false;
  showResMul: boolean = false;
  imgURL: SafeResourceUrl = ""
  imgURL2: SafeResourceUrl = ""
  predVals: string[] = []

  constructor(private service: PredictService) {  }

  ngInit() {
    // this.DetailsForm.controls['Age'].patchValue('0')
    // this.DetailsForm.controls['Gender'].patchValue('0')
    // this.DetailsForm.controls['DepCount'].patchValue('0')
    // this.DetailsForm.controls['MonBook'].patchValue('0')
    // this.DetailsForm.controls['TotRelCnt'].patchValue('0')
    // this.DetailsForm.controls['MonInactCnt'].patchValue('0')
    // this.DetailsForm.controls['CredLim'].patchValue('0')
    // this.DetailsForm.controls['TotRevBal'].patchValue('0')
    // this.DetailsForm.controls['TotAmtChng'].patchValue('0')
    // this.DetailsForm.controls['TotTransAmt'].patchValue('0')
    // this.DetailsForm.controls['TotTransCnt'].patchValue('0')
    // this.DetailsForm.controls['TotCntChng'].patchValue('0')
    // this.DetailsForm.controls['AvgUtil'].patchValue('0')
    // this.DetailsForm.controls['EduLvl'].patchValue('0')
    // this.DetailsForm.controls['Marital'].patchValue('0')
    // this.DetailsForm.controls['IncCat'].patchValue('0')
    // this.DetailsForm.controls['CardCat'].patchValue('0')
    // this.DetailsForm.patchValue()
    this.DetailsForm = new FormGroup({
      Age: new FormControl(0),
      Gender: new FormControl(0),
      DepCount: new FormControl(0),
      MonBook: new FormControl(0),
      TotRelCnt: new FormControl(0),
      MonInactCnt: new FormControl(0),
      CredLim: new FormControl(0),
      TotRevBal: new FormControl(0),
      TotAmtChng: new FormControl(0),
      TotTransAmt: new FormControl(0),
      TotTransCnt: new FormControl(0),
      TotCntChng: new FormControl(0),
      AvgUtil: new FormControl(0),
      EduLvl: new FormControl(0),
      Marital: new FormControl(0),
      IncCat: new FormControl(0),
      CardCat: new FormControl(0),
    });
    console.log("lalalala")
    console.log(this.DetailsForm.value)
  }

  SubmitForm() {
    this.showRes = false;
    this.showResMul = false;
    this.pred = "";
    var fields:any[] = []
    this.imgURL = "";
    // var names = ['Age', 'Gender', 'DepCount', 'MonBook', 'TotRelCnt', 'MonInactCnt', 'CredLim', 'TotRevBal', 'TotAmtChng', 'TotTransAmt', 
    // 'TotTransCnt', 'TotCntChng', 'AvgUtil', 'EduLvl', 'Marital', 'IncCat', 'CardCat']
    // for(const k in names){
    //   // console.log(names[k])
    //   fields.push(this.DetailsForm.controls[names[k]].value)
    // }
    fields.push(this.DetailsForm.controls['Age'].value);
    fields.push(Number(this.DetailsForm.controls['Gender'].value));
    fields.push(Number(this.DetailsForm.controls['DepCount'].value));
    fields.push(this.DetailsForm.controls['MonBook'].value);
    fields.push(Number(this.DetailsForm.controls['TotRelCnt'].value));
    fields.push(Number(this.DetailsForm.controls['MonInactCnt'].value));
    fields.push(this.DetailsForm.controls['CredLim'].value);
    fields.push(this.DetailsForm.controls['TotRevBal'].value);
    fields.push(this.DetailsForm.controls['TotAmtChng'].value);
    fields.push(this.DetailsForm.controls['TotTransAmt'].value);
    fields.push(this.DetailsForm.controls['TotTransCnt'].value);
    fields.push(this.DetailsForm.controls['TotCntChng'].value);
    fields.push(this.DetailsForm.controls['AvgUtil'].value);
    fields.push(Number(this.DetailsForm.controls['EduLvl'].value));
    fields.push(Number(this.DetailsForm.controls['Marital'].value));
    fields.push(Number(this.DetailsForm.controls['IncCat'].value));
    fields.push(Number(this.DetailsForm.controls['CardCat'].value));
    
    console.log(fields);

    this.service.predict(fields).subscribe(data => {
      if(data){
        console.log(data);
        this.showRes = true;
        this.pred = data.text;
        this.service.getPie().subscribe(img => {
          if(img){
            this.imgURL = img
          }
        });

      }
    });
  }

  selectFile(event: any): void {
    this.showRes = false
    this.showResMul = false
    var selectedFiles = event.target.files;
    console.log(selectedFiles)
    const file: File | null = selectedFiles.item(0);
    if(file){
      this.service.upload(file).subscribe({
        next: (event: any) => {
          console.log(event)
          if (event.info == 1) {
            console.log("Files Uploaded");
            this.mul();
          }
        }
      })
    }

  }

  mul(): void {
    this.showResMul = false;
    this.service.test2().subscribe(data =>{
      if(data){
        this.showResMul = true;
        this.predVals = data;
        this.service.getBar().subscribe(img => {
          if(img){
            this.imgURL2 = img
          }
        });
      }
    });
  }

}


// DetailsForm = new FormGroup({
  //   Age: new FormControl('Age'),
  //   Gender: new FormControl('Gender'),
  //   DepCount: new FormControl('DepCount'),
  //   MonBook: new FormControl('MonBook'),
  //   TotRelCnt: new FormControl('TotRelCnt'),
  //   MonInactCnt: new FormControl('MonInactCnt'),
  //   CredLim: new FormControl('CredLim'),
  //   TotRevBal: new FormControl('TotRevBal'),
  //   TotAmtChng: new FormControl('TotAmtChng'),
  //   TotTransAmt: new FormControl('TotTransAmt'),
  //   TotTransCnt: new FormControl('TotTransCnt'),
  //   TotCntChng: new FormControl('TotCntChng'),
  //   AvgUtil: new FormControl('AvgUtil'),
  //   EduLvl: new FormControl('EduLvl'),
  //   Marital: new FormControl('Marital'),
  //   IncCat: new FormControl('IncCat'),
  //   CardCat: new FormControl('CardCat'),
  // });